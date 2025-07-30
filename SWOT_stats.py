import os
from pathlib import Path
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from matplotlib.colors import LogNorm
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize


#Load SWOT Data into GeoDataFrame
file = Path('/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/SWOT_pine_island_final.nc')
ds_raster = xr.open_dataset(file, engine='h5netcdf')

# Load all data variables dynamically
df = pd.DataFrame({var: ds_raster[var].values.ravel() for var in ds_raster.data_vars})

# Add coordinates
df['x'] = ds_raster['x'].values.ravel()
df['y'] = ds_raster['y'].values.ravel()
geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
swot_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3031")

#filter the data for based on the wse_qual values (there is no bad values in the dataset)
gdf_good = swot_gdf[swot_gdf['wse_qual'] == 0]
gdf_suspect = swot_gdf[swot_gdf['wse_qual'] == 1]
gdf_degraded = swot_gdf[swot_gdf['wse_qual'] == 2]

print(f"Good: {len(gdf_good)}, Suspect: {len(gdf_suspect)}, Degraded: {len(gdf_degraded)}")


#calculate RMSE, ubRMSE and Bias
def calc_rmse_and_bias(df, diff_col='elevation_diff'):
    valid = df[diff_col].dropna()
    rmse = np.sqrt(mean_squared_error(np.zeros_like(valid), valid))
    mean_bias = valid.mean()
    unbiased_rmse = np.sqrt(mean_squared_error(np.zeros_like(valid), valid - mean_bias))
    return rmse, unbiased_rmse, mean_bias

for label, data in [('All', swot_gdf), ('Good', gdf_good), ('Suspect', gdf_suspect), ('Degraded', gdf_degraded)]:
    rmse, unbiased_rmse, bias = calc_rmse_and_bias(data)
    print(f"{label}: RMSE={rmse:.3f}, Unbiased RMSE={unbiased_rmse:.3f}, Bias={bias:.3f}")


#Load CryoSat-2 Data to perform correlation with SWOT data 
cryosat_path = Path('/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/gridded_cryosat_250m/Cryosat_gridded_elevation_250m.tif')
with rasterio.open(cryosat_path) as src:
    cryosat_band = src.read(1)
    transform = src.transform
    xs, ys, zs = [], [], []

    for row in range(cryosat_band.shape[0]):
        for col in range(cryosat_band.shape[1]):
            val = cryosat_band[row, col]
            if not np.isnan(val):
                x, y = rasterio.transform.xy(transform, row, col)
                xs.append(x)
                ys.append(y)
                zs.append(val)
cryosat_df = pd.DataFrame({'x': xs, 'y': ys, 'cryosat_elevation': zs})

coords_swot = np.column_stack([swot_gdf['x'], swot_gdf['y']])
coords_cryo = np.column_stack([cryosat_df['x'], cryosat_df['y']])
tree = cKDTree(coords_cryo)
dist, idx = tree.query(coords_swot, k=1)
matched_cryosat = cryosat_df.iloc[idx]['cryosat_elevation'].values

valid = ~np.isnan(swot_gdf['ellipsoid_height'].values) & ~np.isnan(matched_cryosat)
swot_vals = swot_gdf['ellipsoid_height'].values[valid]
cryosat_vals = matched_cryosat[valid]

slope, intercept, r_value, *_ = linregress(cryosat_vals, swot_vals)

# Plot the correlation realtionship between CryoSat and SWOT data
plt.figure(figsize=(10, 6))
hist, xedges, yedges = np.histogram2d(cryosat_vals, swot_vals, bins=200)
hist = np.where(hist == 0, np.nan, hist)
plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', norm=LogNorm())
plt.plot(np.linspace(*plt.xlim(), 250), slope * np.linspace(*plt.xlim(), 250) + intercept, color='red', label=f'Best fit (r={r_value:.3f})')
plt.xlabel('CryoSat Elevation (m)')
plt.ylabel('SWOT Height (m)')
plt.title('SWOT Height vs CryoSat Elevation')
plt.colorbar(label='Counts (log scale)')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Correlation (r) = {r_value:.3f}")


# Random Forest Regression to determine feature importance
# Adjust this to use other elevation difference targets if needed
target = 'elevation_diff'  # or 'elevation_diff_icesat', 'elevation_diff_rema'

exclude_cols = ['geometry', 'x', 'y', 'wse', 'ellipsoid_height', 'wse_qual', 'granule_number', 'pole_tide', 'solid_earth_tide', 'load_tide_got', 'load_tide_fes']
features = [col for col in swot_gdf.columns if col not in exclude_cols and col != target]

X = swot_gdf[features].fillna(0)
y = swot_gdf[target].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

# Makes a simple plot of the importances derived from the Random Forest model
plt.figure(figsize=(12, 8))
importances.plot(kind='bar', color='steelblue')
plt.title(f'Random Forest Importance: Predicting {target}')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()



