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
from matplotlib.colors import TwoSlopeNorm
from ew_common.bounding_box import BoundingBox
from ew_common.raster_tools.raster_np import RasterNp
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load SWOT Combined Dataframe 
def load_geodataframe_swot_raster(file_path: str, granule_number=None, engine='h5netcdf') -> gpd.GeoDataFrame:
    ds = xr.open_dataset(file_path, engine=engine)
    df = pd.DataFrame({
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
        'x': ds.x.values.ravel().astype('float64'),
        'y': ds.y.values.ravel().astype('float64'),
        'ellipsoid_height': ds.ellipsoid_height.values.ravel().astype('float32'),
        'cross_track': ds.cross_track.values.ravel().astype('float32')
    })

    geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3031")


def remove_outliers(gdf: gpd.GeoDataFrame, column: str, lower_q=0.01, upper_q=0.99) -> gpd.GeoDataFrame:
    """Trim outliers outside specified quantiles."""
    lower = gdf[column].quantile(lower_q)
    upper = gdf[column].quantile(upper_q)
    return gdf[(gdf[column] >= lower) & (gdf[column] <= upper)]

def filter_by_cross_track(gdf: gpd.GeoDataFrame, min_dist=10000, max_dist=60000) -> gpd.GeoDataFrame:
    """Filter rows by cross-track distance."""
    return gdf[(np.abs(gdf['cross_track']) >= min_dist) & (np.abs(gdf['cross_track']) <= max_dist)]



# Interpolation Utilities from Earthwave

def grid_interpolation(raster_file_path: str, xs: List[float], ys: List[float],
                       bounding_box: BoundingBox = None) -> np.ndarray:
    raster_df = RasterNp.load_from_tiff(raster_file_path, bounding_box=bounding_box).convert_to_dataframe("z")
    valid_mask = np.isfinite(raster_df.z)
    xy_dem = np.column_stack((raster_df.x[valid_mask], raster_df.y[valid_mask]))
    values = raster_df.z[valid_mask]
    return griddata(xy_dem, values, np.column_stack((xs, ys)), method="cubic")


def grid_bivariate_interpolation(raster_file_path: str, xs: List[float], ys: List[float],
                                 bounding_box: BoundingBox = None, allow_nans=False) -> np.ndarray:
    raster = RasterNp.load_from_tiff(raster_file_path, bounding_box)
    x, y, z = raster.get_centre_points()

    if allow_nans:
        mask = np.isnan(z)
        z[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), z[~mask])

    fitted = RectBivariateSpline(x, y, z, s=0)
    return fitted.ev(xs, ys)


def grid_bivariate_difference(raster_file_path: str, data: pd.DataFrame, param: str,
                              allow_nans: bool = False, bounding_box: BoundingBox = None) -> np.ndarray:
    interp_vals = grid_bivariate_interpolation(raster_file_path, data.x.values, data.y.values,
                                               bounding_box=bounding_box, allow_nans=allow_nans)
    return data[param] - interp_vals



if __name__ == "__main__":
    # File paths
    folder_path = Path('/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/cleaned_mosaicked_dataframes')
    swot_files = [f for f in folder_path.glob("*.nc")]

    # Load all SWOT GeoDataFrames
    gdf_list = [load_geodataframe_swot_raster(f) for f in swot_files]
    swot_gdf = pd.concat(gdf_list, ignore_index=True)
    swot_gdf = gpd.GeoDataFrame(swot_gdf, geometry='geometry', crs="EPSG:3031")
    print(f"Loaded {len(swot_gdf)} total SWOT points")

    # Remove ellipsoid height outliers
    gdf_trimmed = remove_outliers(swot_gdf, column='ellipsoid_height')
    print(f"After trimming: {len(gdf_trimmed)} points remain")
   
    # Filter by cross-track distance
    gdf_trimmed = filter_by_cross_track(gdf_trimmed)


    # Elevation difference with CryoSat DEM
    cryosat_raster_path = Path(
        "/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/gridded_cryosat_250m/Cryosat_gridded_elevation_250m.tif"
    )
    gdf_trimmed["elevation_diff"] = grid_bivariate_difference(
        raster_file_path=str(cryosat_raster_path),
        data=gdf_trimmed,
        param="ellipsoid_height",
        allow_nans=True
    )

    # Elevation difference with REMA DEM
    rema_raster_path = Path(
        "/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/rema_mosaic/REMA_Mosaic_Pineisland.tif"
    )
    gdf_trimmed["elevation_diff_rema"] = grid_bivariate_difference(
        raster_file_path=str(rema_raster_path),
        data=gdf_trimmed,
        param="ellipsoid_height",
        allow_nans=True
    )

    # Elevation difference with ICESat-2 DEM
    icesat_raster_path = Path(
        "/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/gridded_icesat_1000m/ICESat_Gridded_Elevation.tif"
    )
    gdf_trimmed["elevation_diff_icesat"] = grid_bivariate_difference(
        raster_file_path=str(icesat_raster_path),
        data=gdf_trimmed,
        param="ellipsoid_height",
        allow_nans=True
    )
    #inspect the data 
    print(gdf_trimmed.head())

    #plotting the elevation difference data with the 2021 PIG grounding line and a basemap from sentinel-1

    # Load grounding line shapefile
    grounding_line_shp = '/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/grounding_line/20210305000000_AIS_CCI_GLL.2.0_20210305/20210305000000_AIS_CCI_GLL.2.0_20210305.shp'
    grounding_line_gdf = gpd.read_file(grounding_line_shp)

    #Load Sentinel-1 raster image
    sentinel_1_tiff_path = '/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/Sentinel-1_tiffs/Sentinel_1_15_04_25_3031.tif'

    with rasterio.open(sentinel_1_tiff_path) as src:
        sentinel_1_img = src.read(1)
        sentinel_1_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        sentinel_1_crs = src.crs
        sentinel_transform = src.transform

    #Ensure correct CRS for all GeoDataFrames
    target_crs = "EPSG:3031"

    if gdf_trimmed.crs is not None and gdf_trimmed.crs.to_string() != target_crs:
        gdf_trimmed = gdf_trimmed.to_crs(target_crs)
    if grounding_line_gdf.crs is not None and grounding_line_gdf.crs.to_string() != target_crs:
        grounding_line_gdf = grounding_line_gdf.to_crs(target_crs)

    #WSE color scale limits
    wse_min = gdf_trimmed['elevation_diff'].min()
    wse_max = gdf_trimmed['elevation_diff'].max()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot Sentinel-1 background
    ax.imshow(
        sentinel_1_img,
        extent=sentinel_1_extent,
        origin='upper', 
        cmap='gray',
        alpha=0.8,
        zorder=0
    )
    
    # Plot SWOT elevation difference
    sc = ax.scatter(
        gdf_trimmed['x'],
        gdf_trimmed['y'],
        c=gdf_trimmed['elevation_diff'], #adjust as necessary dependent on either plotting elevation difference between  CryoSat, ICESat-2 or REMA
        cmap='RdYlBu',
        s=1,
        alpha=0.7,
        norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=60)
    )
    
    # Plot grounding line
    grounding_line_gdf.plot(ax=ax, color='black', linewidth=1, label='2021 PIG Grounding Line')
    
    # Title, labels, colorbar
    cbar = fig.colorbar(sc, ax=ax, label='Elevation Difference (m)', shrink=0.9)
    ax.set_title('Elevation Difference (m) between SWOT and CryoSat-2 measurements', fontsize=12)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_aspect('equal')
    
    # Set plot extent
    buffer = 3000
    x_min, x_max = gdf_trimmed['x'].min(), gdf_trimmed['x'].max()
    y_min, y_max = gdf_trimmed['y'].min(), gdf_trimmed['y'].max()
    ax.set_xlim(x_min, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    
    # Show legend and final plot
    ax.legend()
    plt.show()
