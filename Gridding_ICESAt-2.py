import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin

def process_icesat_to_geotiff(
    icesat_folder: str | Path,
    output_path: str | Path,
    resolution: int = 1000,
    quality_flag: int = 0,
    crs_epsg: int = 3031,
    verbose: bool = True
):
    """
    Load ATL06 ICESat-2 data from all beams, filter by quality, grid by median, and export to GeoTIFF.

    Parameters:
    - icesat_folder: Path to folder with ATL06 .h5 files.
    - output_path: Path for output GeoTIFF file.
    - resolution: Grid resolution in meters.
    - quality_flag: ATL06 quality filter (default: 0 = good).
    - crs_epsg: Target projection (default EPSG:3031).
    - verbose: Whether to print progress.
    """
    icesat_folder = Path(icesat_folder)
    all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    files = list(icesat_folder.glob("*.h5"))

    gdf_list = []

    for fpath in files:
        if verbose:
            print(f"Processing {fpath.name}")
        try:
            with h5py.File(fpath, 'r') as f:
                for beam in all_beams:
                    if beam not in f:
                        continue
                    try:
                        group = f[f"{beam}/land_ice_segments"]
                        quality = group['atl06_quality_summary'][:]
                        valid = quality == quality_flag
                        if not np.any(valid):
                            continue

                        h_li = group['h_li'][valid]
                        lat = group['latitude'][valid]
                        lon = group['longitude'][valid]

                        df = pd.DataFrame({
                            'h_li': h_li,
                            'latitude': lat,
                            'longitude': lon
                        })

                        gdf = gpd.GeoDataFrame(
                            df,
                            geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                            crs='EPSG:4326'
                        ).to_crs(epsg=crs_epsg)

                        gdf['x'] = gdf.geometry.x
                        gdf['y'] = gdf.geometry.y
                        gdf_list.append(gdf)

                    except Exception as e:
                        print(f"Beam error in {beam}: {e}")
        except Exception as e:
            print(f"Could not open file {fpath.name}: {e}")

    if not gdf_list:
        print("No data found.")
        return

    # Combine all
    gdf_all = pd.concat(gdf_list, ignore_index=True)

    # Round x/y to grid
    gdf_all['x_round'] = (gdf_all['x'] // resolution) * resolution
    gdf_all['y_round'] = (gdf_all['y'] // resolution) * resolution

    # Aggregate by median
    grouped = gdf_all.groupby(['x_round', 'y_round'])['h_li'].median().reset_index()

    # Create 2D raster grid
    xs = np.sort(grouped['x_round'].unique())
    ys = np.sort(grouped['y_round'].unique())[::-1]  # flip y for raster

    x_map = {x: i for i, x in enumerate(xs)}
    y_map = {y: i for i, y in enumerate(ys)}

    raster = np.full((len(ys), len(xs)), np.nan, dtype='float32')

    for _, row in grouped.iterrows():
        x_idx = x_map[row['x_round']]
        y_idx = y_map[row['y_round']]
        raster[y_idx, x_idx] = row['h_li']

    # Export GeoTIFF file
    transform = from_origin(xs[0], ys[0] + resolution, resolution, resolution) #resolution is 1000m for exporting to the geotiff

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype='float32',
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(raster, 1)

    print(f"Exported gridded GeoTIFF: {output_path}")
