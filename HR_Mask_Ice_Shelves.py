import fiona 
import geopandas as gpd
import pandas as pd

#load in the kml file
file = ('swot_science_hr_Nov2024-v09-seasonal_perPass.kml')

gdf_list = []

for layer in fiona.listlayers(file):
    gdf = gpd.read_file(file, driver='LIBKML', layer=layer)
    gdf_list.append(gdf)

gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# Save the GeoDataFrame to a GeoPackage file
gdf.to_file("/data/ox1/working/cryotempo_investigations/swot_example_data/oscar_test_data/data_pineisland_1/HR_Mask/hr_mask_ant_iceshelves.gpkg", driver="GPKG")
