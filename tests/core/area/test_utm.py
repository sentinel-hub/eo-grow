import pytest
from geopandas import GeoDataFrame
from shapely import Polygon

from sentinelhub import CRS, Geometry

from eogrow.core.area import UtmZoneAreaManager
from eogrow.core.area.utm import create_utm_zone_grid

LARGE_AREA_CONFIG = {
    "geometry_filename": "test_large_area.geojson",
    "patch": {"size_x": 1000000, "size_y": 1000000, "buffer_x": 0, "buffer_y": 0},
}


AREA_CONFIG = {
    "geometry_filename": "test_area.geojson",
    "patch": {"size_x": 2400, "size_y": 1100, "buffer_x": 120, "buffer_y": 55},
}


@pytest.mark.parametrize(
    ("config", "expected_zone_num", "expected_bbox_num"),
    [
        (AREA_CONFIG, 1, 2),
        (LARGE_AREA_CONFIG, 71, 368),
    ],
)
def test_bbox_split(storage, config, expected_zone_num, expected_bbox_num):
    area_manager = UtmZoneAreaManager.from_raw_config(config, storage)

    grid = area_manager.get_grid()

    assert isinstance(grid, dict)
    assert len(grid) == expected_zone_num

    bbox_count = 0
    for crs, subgrid in grid.items():
        assert crs == CRS(subgrid.crs)
        assert isinstance(subgrid, GeoDataFrame)
        bbox_count += len(subgrid.index)

    assert bbox_count == expected_bbox_num


def test_cache_name(storage):
    area_manager = UtmZoneAreaManager.from_raw_config(AREA_CONFIG, storage)

    assert area_manager.get_grid_cache_filename() == "UtmZoneAreaManager_test_area_2400_1100_120.0_55.0_0.0_0.0.gpkg"


@pytest.mark.parametrize(
    ("offset_x", "offset_y", "buffer_x", "buffer_y", "polygon_first", "polygon_last"),
    [
        (
            0,
            0,
            0,
            0,
            Polygon([[710000, 5305000], [710000, 5310000], [715000, 5310000], [715000, 5305000], [710000, 5305000]]),
            Polygon([[285000, 5325000], [285000, 5330000], [290000, 5330000], [290000, 5325000], [285000, 5325000]]),
        ),
        (
            10,
            20,
            30,
            40,
            Polygon([[709980, 5304980], [709980, 5310060], [715040, 5310060], [715040, 5304980], [709980, 5304980]]),
            Polygon([[284980, 5324980], [284980, 5330060], [290040, 5330060], [290040, 5324980], [284980, 5324980]]),
        ),
    ],
)
def test_create_utm_zone_grid(offset_x, offset_y, buffer_x, buffer_y, polygon_first, polygon_last):
    # Covers UTM zones 33N and 34N
    geometry = Geometry(
        "POLYGON ((17.87338 47.88549, 17.87338 48.06064, 18.14598 48.06064, 18.14598 47.88549, 17.87338 47.88549))",
        crs=CRS.WGS84,
    )
    grid = create_utm_zone_grid(geometry, "eopatch_name", (5000, 5000), (offset_x, offset_y), (buffer_x, buffer_y))

    assert isinstance(grid, dict)
    assert len(grid) == 2
    assert set(grid.keys()) == {CRS("32633"), CRS("32634")}

    n_expected_tiles = 15
    for gdf in grid.values():
        assert isinstance(gdf, GeoDataFrame)
        assert set(gdf.columns) == {"id", "eopatch_name", "geometry"}
        assert len(gdf.index) == n_expected_tiles

    # explicit check for a few tiles
    assert grid[CRS("32633")].iloc[0].to_dict() == {
        "id": 0,
        "eopatch_name": "eopatch-id-00-col-0-row-0",
        "geometry": polygon_first,
    }
    assert grid[CRS("32634")].iloc[-1].to_dict() == {
        "id": 29,
        "eopatch_name": "eopatch-id-29-col-2-row-4",
        "geometry": polygon_last,
    }
