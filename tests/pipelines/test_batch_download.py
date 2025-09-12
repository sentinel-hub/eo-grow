from geopandas import GeoDataFrame

from sentinelhub import CRS, Geometry

from eogrow.core.area.utm import create_utm_zone_grid
from eogrow.pipelines.download_batch import to_batch_grid_format


def test_to_batch_grid_format():
    # Covers UTM zones 33N and 34N
    area_geometry = Geometry(
        "POLYGON ((17.87338 47.88549, 17.87338 48.06064, 18.14598 48.06064, 18.14598 47.88549, 17.87338 47.88549))",
        crs=CRS.WGS84,
    )

    bbox_size = (5000, 5000)
    bbox_offset = (0, 0)
    bbox_buffer = (0, 0)
    image_size = (512, 512)
    resolution = 10

    grid = create_utm_zone_grid(area_geometry, "identifier", bbox_size, bbox_offset, bbox_buffer)
    grid = to_batch_grid_format(grid, image_size, resolution)

    assert isinstance(grid, dict)
    assert len(grid) == 2
    assert set(grid.keys()) == {CRS("32633"), CRS("32634")}

    n_expected_tiles = 15
    for gdf in grid.values():
        assert isinstance(gdf, GeoDataFrame)
        assert set(gdf.columns) == {"id", "identifier", "geometry", "width", "height", "resolution"}
        assert len(gdf.index) == n_expected_tiles
        assert all(gdf["width"] == image_size[0])
        assert all(gdf["height"] == image_size[1])
        assert all(gdf["resolution"] == resolution)
