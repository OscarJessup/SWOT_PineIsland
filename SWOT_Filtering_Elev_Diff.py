import os
import re
import math
from pathlib import Path
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from shapely.geometry import Point
from scipy.interpolate import griddata, RectBivariateSpline
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from ew_common.bounding_box import BoundingBox
from ew_common.raster_tools.raster_np import RasterNp

# Load SWOT data as GeoDataFrame
def load_geodataframe_swot_raster(file_path: str, engine='h5netcdf') -> gpd.GeoDataFrame:
    ds = xr.open_dataset(file_path, engine=engine)
    
    #Load all data variables
    df = pd.DataFrame({var: ds[var].values.ravel() for var in ds.data_vars})
    
    # Add x and y coordinates
    if 'x' not in df.columns:
        df['x'] = ds['x'].values.ravel()
    if 'y' not in df.columns:
        df['y'] = ds['y'].values.ravel()

    
    for col in df.columns:
        if col not in ['x', 'y']:
            df[col] = df[col].astype('float32')

    # Construct geometry from x and y coordinates
    geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3031")

# Remove statistical outliers
def remove_outliers(gdf: gpd.GeoDataFrame, column: str, lower_q=0.01, upper_q=0.99) -> gpd.GeoDataFrame:
    lower = gdf[column].quantile(lower_q)
    upper = gdf[column].quantile(upper_q)
    return gdf[(gdf[column] >= lower) & (gdf[column] <= upper)]

# Filter by cross-track range
def filter_by_cross_track(gdf: gpd.GeoDataFrame, min_dist=10000, max_dist=60000) -> gpd.GeoDataFrame:
    return gdf[(np.abs(gdf['cross_track']) >= min_dist) & (np.abs(gdf['cross_track']) <= max_dist)]

# Cubic grid interpolation from raster
def grid_interpolation(raster_file_path: str, xs: List[float], ys: List[float],
                       bounding_box: BoundingBox = None) -> np.ndarray:
    raster_df = RasterNp.load_from_tiff(raster_file_path, bounding_box=bounding_box).convert_to_dataframe("z")
    valid_mask = np.isfinite(raster_df.z)
    xy_dem = np.column_stack((raster_df.x[valid_mask], raster_df.y[valid_mask]))
    values = raster_df.z[valid_mask]
    return griddata(xy_dem, values, np.column_stack((xs, ys)), method="cubic")

# Bivariate spline interpolation
def grid_bivariate_interpolation(raster_file_path: str, xs: List[float], ys: List[float],
                                 bounding_box: BoundingBox = None, allow_nans=False) -> np.ndarray:
    raster = RasterNp.load_from_tiff(raster_file_path, bounding_box)
    x, y, z = raster.get_centre_points()

    if allow_nans:
        mask = np.isnan(z)
        z[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), z[~mask])

    fitted = RectBivariateSpline(x, y, z, s=0)
    return fitted.ev(xs, ys)

# Elevation difference between SWOT and a given DEM
def grid_bivariate_difference(raster_file_path: str, data: pd.DataFrame, param: str,
                              allow_nans: bool = False, bounding_box: BoundingBox = None) -> np.ndarray:
    interp_vals = grid_bivariate_interpolation(raster_file_path, data.x.values, data.y.values,
                                               bounding_box=bounding_box, allow_nans=allow_nans)
    return data[param] - interp_vals

if __name__ == "__main__":
    # File paths
    folder_path = Path('/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/cleaned_mosaicked_dataframes')
    swot_files = list(folder_path.glob("*.nc"))

    # Load SWOT data
    gdf_list = [load_geodataframe_swot_raster(f) for f in swot_files]
    swot_gdf = pd.concat(gdf_list, ignore_index=True)
    swot_gdf = gpd.GeoDataFrame(swot_gdf, geometry='geometry', crs="EPSG:3031")
    print(f"Loaded {len(swot_gdf)} total SWOT points")

    # Remove outliers
    gdf_trimmed = remove_outliers(swot_gdf, column='ellipsoid_height')
    print(f"After trimming: {len(gdf_trimmed)} points remain")

    # Cross-track filter
    gdf_trimmed = filter_by_cross_track(gdf_trimmed)

    # DEM difference: CryoSat-2
    cryosat_raster_path = Path("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/gridded_cryosat_250m/Cryosat_gridded_elevation_250m.tif")
    gdf_trimmed["elevation_diff"] = grid_bivariate_difference(str(cryosat_raster_path), gdf_trimmed, "ellipsoid_height", allow_nans=True)

    # DEM difference: REMA
    rema_raster_path = Path("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/rema_mosaic/REMA_Mosaic_Pineisland.tif")
    gdf_trimmed["elevation_diff_rema"] = grid_bivariate_difference(str(rema_raster_path), gdf_trimmed, "ellipsoid_height", allow_nans=True)

    # DEM difference: ICESat-2
    icesat_raster_path = Path("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/gridded_icesat_1000m/ICESat_Gridded_Elevation.tif")
    gdf_trimmed["elevation_diff_icesat"] = grid_bivariate_difference(str(icesat_raster_path), gdf_trimmed, "ellipsoid_height", allow_nans=True)

    # Save the processed DataFrame to a netCDF file
    output_file = Path("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/SWOT_pine_island_final.nc")
    gdf_trimmed.to_netcdf(output_file, engine='h5netcdf', format='NETCDF4', mode='w')
