import earthaccess
auth = earthaccess.login()
raster_results = earthaccess.search_data(
    short_name = 'SWOT_L2_HR_Raster_2.0',
    bounding_box=(),
    temporal =('2024-01-20 12:00:00', '2025-06-24 23:59:59'),
    granule_name = '',  
    count =500
)
print("Number of granules found:", len(raster_results))
for r in raster_results:
    print(r)
for granule in raster_results:
    links = granule.data_links()
    if links:
        earthaccess.download(links, local_path='')
