import os
import re
from pathlib import Path
from typing import Optional

import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# Setup Paths and File Filtering
data_root = Path("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/raster_working_folder")

shelf_dirs = {
    "main": data_root / "Main_Ice_Shelf",
    "north": data_root / "North_Ice_Shelf",
    "south": data_root / "South_Ice_Shelf"
}

valid_cycles = set(str(c) for c in range(27, 34))  

def get_cycle_id(filename: str) -> Optional[str]:
    match = re.search(r"_C(\d{3})_", filename)
    if match:
        return match.group(1).lstrip("0") 
    return None

all_filtered_files = []
for shelf, shelf_path in shelf_dirs.items():
    if shelf_path.exists():
        nc_files = list(shelf_path.glob("*.nc"))
        filtered = [f for f in nc_files if get_cycle_id(f.name) in valid_cycles]
        all_filtered_files.extend(filtered)
    else:
        print(f"Warning: {shelf_path} does not exist.")

print(f"Found {len(all_filtered_files)} SWOT .nc files for cycles 27–34.")


# Load SWOT Raster as GeoDataFrame
def load_swot_raster_to_gdf(file_path, engine='h5netcdf'):
    ds = xr.open_dataset(file_path, engine=engine)
    
    data = {
        'wse': ds.wse.values.ravel().astype('float32'),
        'sig0': ds.sig0.values.ravel().astype('float32'),
        'wse_qual': ds.wse_qual.values.ravel().astype('float32'),
        'geoid': ds.geoid.values.ravel().astype('float32'),
        'model_wet_tropo_cor': ds.model_wet_tropo_cor.values.ravel().astype('float32'),
        'model_dry_tropo_cor': ds.model_dry_tropo_cor.values.ravel().astype('float32'),
        'iono_cor_gim_ka': ds.iono_cor_gim_ka.values.ravel().astype('float32'),
        'height_cor_xover': ds.height_cor_xover.values.ravel().astype('float32'),
        'solid_earth_tide': ds.solid_earth_tide.values.ravel().astype('float32'),
        'pole_tide': ds.pole_tide.values.ravel().astype('float32'),
        'load_tide_fes': ds.load_tide_fes.values.ravel().astype('float32'),
        'sig0_cor_atmos_model': ds.sig0_cor_atmos_model.values.ravel().astype('float32'),
        'load_tide_got': ds.load_tide_got.values.ravel().astype('float32'),
        'layover_impact': ds.layover_impact.values.ravel().astype('float32'),
        'inc': ds.inc.values.ravel().astype('float32'),
        'dark_frac': ds.dark_frac.values.ravel().astype('float32'),
        'n_wse_pix': ds.n_wse_pix.values.ravel().astype('float32'),
        'n_sig0_pix': ds.n_sig0_pix.values.ravel().astype('float32'),
        'water_frac': ds.water_frac.values.ravel().astype('float32'),
        'cross_track': ds.cross_track.values.ravel().astype('float32'),
    }

    gdf = gpd.GeoDataFrame(
        data=data,
        geometry=gpd.points_from_xy(ds.longitude.values.ravel(), ds.latitude.values.ravel()),
        crs="EPSG:4326"
    ).to_crs("EPSG:3031")

    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    return gdf


# Mosaic Function
def mosaic_geodataframes(gdf_list, value_column, prefer='median', precision=4):
    assert prefer in ['min', 'max', 'median'], "Invalid preference. Choose from 'min', 'max', 'median'."

    combined = pd.concat(gdf_list, ignore_index=True).dropna(subset=['x', 'y'])
    combined['x_round'] = combined['x'].round(precision)
    combined['y_round'] = combined['y'].round(precision)

    group_cols = ['x_round', 'y_round']
    
    if prefer == 'median':
        agg_df = combined.groupby(group_cols, dropna=True).median(numeric_only=True).reset_index()
    else:
        idx_func = combined.groupby(group_cols)[value_column].idxmax if prefer == 'max' else combined.groupby(group_cols)[value_column].idxmin
        idx = idx_func()
        agg_df = combined.loc[idx].reset_index(drop=True)

    agg_df['geometry'] = gpd.points_from_xy(agg_df['x_round'], agg_df['y_round'])
    agg_df.drop(columns=['x_round', 'y_round'], inplace=True)
    
    return gpd.GeoDataFrame(agg_df, geometry='geometry', crs=gdf_list[0].crs)

# Add Ellipsoid Height Column
def add_ellipsoid_height(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["ellipsoid_height"] = (
        gdf["wse"]
        + gdf["geoid"]
        + gdf["solid_earth_tide"]
        + gdf["pole_tide"]
        + gdf["load_tide_fes"]
    )
    return gdf
 
# Load, Mosaic, and Process
if __name__ == "__main__":
    print("Loading SWOT raster files...")

    gdf_list = [load_swot_raster_to_gdf(fp) for fp in all_filtered_files]
    print(f"Loaded {len(gdf_list)} GeoDataFrames.")

    gdf_combined = mosaic_geodataframes(gdf_list, value_column='wse', prefer='median')
    print(f"Mosaicked GeoDataFrame has {len(gdf_combined)} rows.")

    gdf_combined = add_ellipsoid_height(gdf_combined)
    print("Ellipsoid height column added.")
