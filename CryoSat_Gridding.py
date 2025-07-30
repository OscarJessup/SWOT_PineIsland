import os
from pathlib import Path
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import numpy as np
import math
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize

folder_path = Path('/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/Cryosat_Swath')
cryosat_file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.nc')]

def load_geodataframe(ds):
    df = pd.DataFrame({
        'elevation': ds.elevation.values.astype('float32'),
        'x': ds.x.values.astype('float32'),
        'y': ds.y.values.astype('float32'),
        'time': ds.time.values.astype('int32')
    })
    geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    return gdf

# Load all datasets into GeoDataFrames
gdf_list = []
for file_path in cryosat_file_paths:
    ds = xr.open_dataset(file_path)
    gdf = load_geodataframe(ds)
    gdf_list.append(gdf)

# Concatenate
cryosat_combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

cryosat_combined_gdf.set_crs(epsg=3031, inplace=True)
print(f"Combined GeoDataFrame has {len(cryosat_combined_gdf)} rows")


#Grid Cryosat data
def grid_data_cryosat(gdf: gpd.GeoDataFrame, column_name: str, grid_resolution: float) -> gpd.GeoDataFrame:
    # Ensure the GeoDataFrame has a CRS set before transforming
    half_grid_res = grid_resolution / 2

    # assign to each point the coordinates of the grid cell it belongs to
    gdf['x_centre'] = [math.floor(x / grid_resolution) * grid_resolution + half_grid_res
                       for x in gdf.geometry.x]
    gdf['y_centre'] = [math.floor(y / grid_resolution) * grid_resolution + half_grid_res
                       for y in gdf.geometry.y]

    # use the pandas.DataFrame method "groupby" to aggregate the points that belong to the same grid cell
    grid_cell_df = gdf.groupby(by=['x_centre', 'y_centre'])[column_name].agg('median').reset_index()

    # construct another DataFrame where for each gridded cell
    gridded_data_gdf = gpd.GeoDataFrame({
        f'median_{column_name}': grid_cell_df[column_name],
        'geometry': [box(x - half_grid_res, y - half_grid_res, x + half_grid_res, y + half_grid_res)
                     for x, y in zip(grid_cell_df.x_centre, grid_cell_df.y_centre)]
    }, crs=gdf.crs)

    return gridded_data_gdf
gridded_cryosat = grid_data_cryosat(cryosat_combined_gdf.copy(), 'elevation', 250)

#Exporting Gridded Cryosat data to geotiff so it can be used to perform grid_bivariate_interpolation
if gridded_cryosat.crs is None:
    gridded_cryosat = gridded_cryosat.set_crs(epsg=4326)
gridded_cryosat = gridded_cryosat.to_crs(epsg=3031)

# Define resolution
resolution = 250

# Define raster extent
minx, miny, maxx, maxy = gridded_cryosat.total_bounds
width = int((maxx - minx) / resolution)
height = int((maxy - miny) / resolution)

# Define transform for raster
transform = from_origin(minx, maxy, resolution, resolution)

# Create (geometry, value) pairs for rasterization
shapes = [
    (geom, value) for geom, value in zip(
        gridded_cryosat.geometry,
        gridded_cryosat['median_elevation']
    ) if value is not None
]

# Rasterize
raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    fill=np.nan,
    transform=transform,
    dtype='float32'
)

# Save to GeoTIFF
output_path = '/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/gridded_cryosat_250m/Cryosat_gridded_elevation_250m.tif'
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    transform=transform,
    nodata=np.nan
) as dst:
    dst.write(raster, 1)

print(f"GeoTIFF saved to: {output_path}")
