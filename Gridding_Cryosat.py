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

# Concatenate into one big GeoDataFrame
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
