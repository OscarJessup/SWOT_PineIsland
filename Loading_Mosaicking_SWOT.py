import os
import re
from pathlib import Path
from typing import Optional, List

import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


data = Path("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/raster_working_folder")
output_dir = data.parent / "cleaned_mosaicked_dataframes"
output_file = output_dir / "combined_swot_granules.nc"

SHELF_DIRS = {
    "main": data / "Main_Ice_Shelf",
    "north": data / "North_Ice_Shelf",
    "south": data / "South_Ice_Shelf"
}
VALID_CYCLES = {str(c) for c in range(27, 34)}  # Cycles 27–34


def get_cycle_id(filename: str) -> Optional[str]:
    """Extracts SWOT cycle ID from filename."""
    match = re.search(r"_C(\d{3})_", filename)
    if match:
        return match.group(1).lstrip("0")
    return None

def load_swot_raster_to_gdf(file_path: Path, engine='h5netcdf') -> gpd.GeoDataFrame:
    """Load SWOT raster NetCDF file and return a GeoDataFrame."""
    ds = xr.open_dataset(file_path, engine=engine)

    # Flatten and convert to DataFrame
    df = ds.to_dataframe().reset_index().dropna(subset=["longitude", "latitude"])
    
    # Convert to GeoDataFrame with EPSG:3031 projection
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3031")
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    
    return gdf

def mosaic_geodataframes(gdf_list: List[gpd.GeoDataFrame], precision: int = 4) -> gpd.GeoDataFrame:
    """Mosaics multiple GeoDataFrames by median value per cell."""
    combined = pd.concat(gdf_list, ignore_index=True).dropna(subset=['x', 'y'])
    combined["x_round"] = combined["x"].round(precision)
    combined["y_round"] = combined["y"].round(precision)

    group_cols = ["x_round", "y_round"]
    agg_df = combined.groupby(group_cols, dropna=True).median(numeric_only=True).reset_index()

    agg_df["geometry"] = gpd.points_from_xy(agg_df["x_round"], agg_df["y_round"])
    agg_df.drop(columns=["x_round", "y_round"], inplace=True)

    return gpd.GeoDataFrame(agg_df, geometry="geometry", crs=gdf_list[0].crs)

def add_ellipsoid_height(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Computes ellipsoid height from tidal corrections and geoid."""
    gdf = gdf.copy()
    required_columns = ["wse", "geoid", "solid_earth_tide", "pole_tide", "load_tide_fes"]
    missing = [col for col in required_columns if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns for ellipsoid height: {missing}")

    gdf["ellipsoid_height"] = (
        gdf["wse"]
        + gdf["geoid"]
        + gdf["solid_earth_tide"]
        + gdf["pole_tide"]
        + gdf["load_tide_fes"]
    )
    return gdf

if __name__ == "__main__":

    all_filtered_files = []
    for shelf, shelf_path in SHELF_DIRS.items():
        if shelf_path.exists():
            nc_files = list(shelf_path.glob("*.nc"))
            filtered = [f for f in nc_files if get_cycle_id(f.name) in VALID_CYCLES]
            all_filtered_files.extend(filtered)
        else:
            print(f"Warning: Shelf directory missing: {shelf_path}")

    print(f"Found {len(all_filtered_files)} valid SWOT NetCDF files.")

    print("Loading data into GeoDataFrames...")
    gdf_list = [load_swot_raster_to_gdf(fp) for fp in all_filtered_files]
    print(f"Loaded {len(gdf_list)} individual GeoDataFrames.")

    print("Mosaicking data using median aggregation...")
    gdf_combined = mosaic_geodataframes(gdf_list)
    print(f"Mosaicked GeoDataFrame contains {len(gdf_combined)} points.")

    print("Calculating ellipsoid height...")
    gdf_combined = add_ellipsoid_height(gdf_combined)

    print("Exporting to NetCDF...")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_nc = gdf_combined.drop(columns="geometry")
    ds_nc = xr.Dataset.from_dataframe(df_nc)
    ds_nc.to_netcdf(output_file, mode='w', format='NETCDF4')

    print(f" NetCDF file saved to: {output_file}")
