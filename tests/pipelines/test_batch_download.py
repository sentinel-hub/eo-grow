from geopandas import GeoDataFrame
from shapely import Polygon

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
    assert set(grid.keys()) == {CRS("32633"), CRS("32634")}

    for gdf, n_expected_tiles in zip(grid.values(), (15, 15)):
        assert isinstance(gdf, GeoDataFrame)
        assert set(gdf.columns) == {"id", "identifier", "geometry", "width", "height", "resolution"}
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf.index) == n_expected_tiles
        assert all(gdf["width"] == image_size[0])
        assert all(gdf["height"] == image_size[1])
        assert all(gdf["resolution"] == resolution)

    # explicit check for a few tiles
    assert grid[CRS("32633")].iloc[0].to_dict() == {
        "id": 0,
        "identifier": "eopatch-id-00-col-0-row-0",
        "height": 512,
        "resolution": 10,
        "width": 512,
        "geometry": Polygon(
            [[710000, 5305000], [710000, 5310000], [715000, 5310000], [715000, 5305000], [710000, 5305000]]
        ),
    }
    assert grid[CRS("32634")].iloc[-1].to_dict() == {
        "id": 29,
        "identifier": "eopatch-id-29-col-2-row-4",
        "height": 512,
        "resolution": 10,
        "width": 512,
        "geometry": Polygon(
            [[285000, 5325000], [285000, 5330000], [290000, 5330000], [290000, 5325000], [285000, 5325000]]
        ),
    }
