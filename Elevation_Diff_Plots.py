import os
from pathlib import Path
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib_scalebar.scalebar import ScaleBar
import rasterio


# Load SWOT Data
swot_path = Path('/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/SWOT_pine_island_final.nc')
ds = xr.open_dataset(swot_path, engine='h5netcdf')

df = pd.DataFrame({var: ds[var].values.ravel() for var in ds.data_vars})
df['x'] = ds.x.values.ravel()
df['y'] = ds.y.values.ravel()
df['granule_number'] = ds.granule_number.values.ravel().astype('int32')

geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
swot_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3031")

#2024 Grounding Line
grounding_line_gdf = gpd.read_file(
    '/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/Rignot_Grounding_Line.shp'
).to_crs("EPSG:3031")

#Load Sentinel-1 tif
sentinel_path = '/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/Sentinel-1_tiffs/Sentinel_1_15_04_25_3031.tif'
with rasterio.open(sentinel_path) as src:
    sentinel_img = src.read(1)
    sentinel_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    sentinel_crs = src.crs


#function to plot granule data
def plot_granule(gdf, granule, color_col, cmap, norm, title_suffix, cbar_label):
    group = gdf[gdf['granule_number'] == granule]
    fig, ax = plt.subplots(figsize=(10, 10))

    # Sentinel background
    ax.imshow(sentinel_img, extent=sentinel_extent, origin='upper', cmap='gray', alpha=0.8, zorder=0)

    # Scatter data
    sc = ax.scatter(group['x'], group['y'], c=group[color_col], cmap=cmap, s=1, alpha=0.7, norm=norm)

    # Grounding line
    grounding_line_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1, label='2024 Grounding Line', zorder=2)

    # Title and labels
    ax.set_title(f'Granule {granule}: {title_suffix}', fontsize=12)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_aspect('equal')

    # Extent
    buffer = 3000
    x_min, x_max = group['x'].min(), group['x'].max()
    y_min, y_max = group['y'].min(), group['y'].max()
    ax.set_xlim(x_min, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)

    # Format and legend
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(-2, 2))
    fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
    ax.add_artist(ScaleBar(dx=1, units="m", location='lower left', scale_loc='bottom', length_fraction=0.5, box_alpha=0.5))
    ax.legend()
    plt.tight_layout()
    plt.show()


#Elevation Difference between SWOT and CryoSat-2
for granule in swot_gdf['granule_number'].unique():
    plot_granule(
        gdf=swot_gdf,
        granule=granule,
        color_col='elevation_diff',
        cmap='RdYlBu',
        norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=60),
        title_suffix='Elevation Difference (SWOT - CryoSat-2)',
        cbar_label='Elevation Difference (m)'
    )

#Elevation Difference between SWOT and ICESat-2
for granule in swot_gdf['granule_number'].unique():
    plot_granule(
        gdf=swot_gdf,
        granule=granule,
        color_col='elevation_diff',
        cmap='RdYlBu',
        norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=60),
        title_suffix='Elevation Difference (SWOT - ICESat-2)',
        cbar_label='Elevation Difference (m)'
    )

#Elevation Difference between SWOT and REMA
for granule in swot_gdf['granule_number'].unique():
    plot_granule(
        gdf=swot_gdf,
        granule=granule,
        color_col='elevation_diff',
        cmap='RdYlBu',
        norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=60),
        title_suffix='Elevation Difference (SWOT - REMA)',
        cbar_label='Elevation Difference (m)'
    )

#WSE where Elevation Diff within ±5m between SWOT and CryoSat-2
filtered_gdf = swot_gdf[swot_gdf['elevation_diff'].abs() <= 5]

for granule in filtered_gdf['granule_number'].unique():
    plot_granule(
        gdf=filtered_gdf,
        granule=granule,
        color_col='wse',
        cmap='RdYlBu',
        norm=Normalize(vmin=-30, vmax=130),
        title_suffix='WSE for Elevation Diff ±5m',
        cbar_label='Water Surface Elevation (m)'
    )

