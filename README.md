# SWOT_PineIsland
A repository of code evaluating ice surface elevation heights from SWOT (Surface Water and Ocean Topography Satellite) over the Pine Island Ice Shelf.

SWOT Raster Data Uses the EarthAccess Module to download data directly from https://search.earthdata.nasa.gov/search/granules?portal=podaac-cloud&p=C2799438271-POCLOUD&pg[0][v]=f&pg[0][gsk]=-start_date&fi=KaRIn&tl=1712130367.628!4!!

CryoSat-2 Swath Data is from https://cs2eo.org/

ICESat-2 ATL06 is directly available from https://nsidc.org/data/atl06/versions/6

REMA Data is available from https://rema.apps.pgc.umn.edu/ with REMA mosaic exporting as a GeoTIFF

To perform elevation difference is it necessary to clone Earthwave's ew_gdal_helpers repository and activate the ew_gdal_helpers environment in conda. 

