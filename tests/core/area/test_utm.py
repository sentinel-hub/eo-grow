import pytest
from geopandas import GeoDataFrame

from sentinelhub import CRS

from eogrow.core.area import UtmZoneAreaManager


@pytest.fixture(scope="session", name="large_area_config")
def large_area_config_fixture():
    return {
        "geometry_filename": "test_large_area.geojson",
        "patch": {"size_x": 1000000, "size_y": 1000000, "buffer_x": 0, "buffer_y": 0},
    }


@pytest.fixture(scope="session", name="area_config")
def area_config_fixture():
    return {
        "geometry_filename": "test_area.geojson",
        "patch": {"size_x": 2400, "size_y": 1100, "buffer_x": 120, "buffer_y": 55},
    }


@pytest.mark.parametrize(
    ("config", "expected_zone_num", "expected_bbox_num"),
    [
        (pytest.lazy_fixture("area_config"), 1, 2),
        (pytest.lazy_fixture("large_area_config"), 71, 368),
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


def test_cache_name(storage, area_config):
    area_manager = UtmZoneAreaManager.from_raw_config(area_config, storage)

    assert area_manager.get_grid_cache_filename() == "UtmZoneAreaManager_test_area_2400_1100_120.0_55.0_0.0_0.0.gpkg"
