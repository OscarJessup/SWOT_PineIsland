import earthaccess
auth = earthaccess.login()
raster_results = earthaccess.search_data(
    short_name = 'SWOT_L2_HR_Raster_2.0',
    #bounding_box=(-104.79089493,-76.13200356,-100.80358706,-75.08083957),
    temporal =('2024-01-20 12:00:00', '2025-06-24 23:59:59'),
    granule_name = '*100m*UTM13C*166_148F*',  
    count =500
)
print("Number of granules found:", len(raster_results))
[display(r) for r in raster_results]
for granule in raster_results:
    links = granule.data_links()
    if links:
        earthaccess.download(links, local_path='')
