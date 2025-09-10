from geopandas import GeoDataFrame

from sentinelhub import CRS, Geometry

from eogrow.pipelines.download_batch import create_batch_grid


def test_create_batch_grid():
    area_geometry = Geometry(
        "POLYGON ((17.87338 47.88549, 17.87338 48.06064, 18.14598 48.06064, 18.14598 47.88549, 17.87338 47.88549))",
        crs=CRS.WGS84,
    )  # Covers UTM zones 33N and 34N

    bbox_size = (5000, 5000)
    bbox_offset = (0, 0)
    bbox_buffer = (0, 0)
    image_size = (512, 512)
    resolution = 10

    grid = create_batch_grid(area_geometry, bbox_size, bbox_offset, bbox_buffer, image_size, resolution)

    assert isinstance(grid, dict)
    assert len(grid) == 2
    assert CRS("32633") in grid
    assert CRS("32634") in grid

    for gdf, n_expected_tiles in zip(grid.values(), (15, 15)):
        assert isinstance(gdf, GeoDataFrame)
        assert set(gdf.columns) == {"id", "identifier", "geometry", "width", "height", "resolution"}
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf.index) == n_expected_tiles
        assert all(gdf["width"] == image_size[0])
        assert all(gdf["height"] == image_size[1])
        assert all(gdf["resolution"] == resolution)
