from unittest.mock import patch

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from sentinelhub import CRS

from eogrow.core.area import NewUtmZoneAreaManager
from eogrow.utils.vector import count_points

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="large_area_config")
def large_area_config_fixture():
    return {
        "area": {"filename": "test_large_area.geojson", "buffer": 1},
        "patch": {"size_x": 1000000, "size_y": 1000000, "buffer_x": 0, "buffer_y": 0},
    }


@pytest.fixture(scope="session", name="area_config")
def area_config_fixture():
    return {
        "area": {"filename": "test_area.geojson", "buffer": 0.001},
        "patch": {"size_x": 2400, "size_y": 1100, "buffer_x": 120, "buffer_y": 55},
    }


@pytest.mark.parametrize(
    "simplification_factor,expected_point_count", [(0, 128), (0.00001, 64), (0.0001, 25), (0.001, 10), (0.1, 5)]
)
def test_get_area_geometry(area_config, storage, simplification_factor, expected_point_count):
    area_config["area"]["simplification_factor"] = simplification_factor
    area_manager = NewUtmZoneAreaManager.from_raw_config(area_config, storage)

    geometry = area_manager.get_area_geometry()
    assert count_points(geometry.geometry) == expected_point_count


@pytest.mark.parametrize(
    "config, expected_zone_num, expected_bbox_num",
    [
        (pytest.lazy_fixture("area_config"), 1, 2),
        (pytest.lazy_fixture("large_area_config"), 71, 368),
    ],
)
def test_bbox_split(storage, config, expected_zone_num, expected_bbox_num):
    area_manager = NewUtmZoneAreaManager.from_raw_config(config, storage)

    grid = area_manager.get_grid()

    _check_area_grid(grid, expected_zone_num, expected_bbox_num)


def test_get_grid_caching(storage, area_config):
    """Tests that subsequent calls use the cached version of the grid. NOTE: implementation specific test!"""
    area_manager = NewUtmZoneAreaManager.from_raw_config(area_config, storage)

    assert area_manager.get_grid_cache_filename() == "NewUtmZoneAreaManager_test_area_2400_1100_120.0_55.0_0.0_0.0.gpkg"

    grid1 = area_manager.get_grid()
    # the second call should not create a new split
    with patch.object(NewUtmZoneAreaManager, "_create_grid") as creation_mock:
        grid2 = area_manager.get_grid()
        creation_mock.assert_not_called()

    for gdf1, gdf2 in zip(grid1.values(), grid2.values()):
        # could potentially fail if the loaded grid doesn't have the same order :/
        assert_geodataframe_equal(gdf1, gdf2, check_index_type=False, check_dtype=False)


def _check_area_grid(grid, expected_zone_num, expected_bbox_num):
    assert isinstance(grid, dict)
    assert len(grid) == expected_zone_num

    bbox_count = 0
    for crs, subgrid in grid.items():
        assert crs == CRS(subgrid.crs)
        assert isinstance(subgrid, GeoDataFrame)
        bbox_count += len(subgrid.index)

    assert bbox_count == expected_bbox_num
