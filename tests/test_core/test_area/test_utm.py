import os

import mock
import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from sentinelhub import BBox, Geometry

from eogrow.core.area import UtmZoneAreaManager
from eogrow.core.config import interpret_config_from_path
from eogrow.utils.vector import count_points

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="large_area_config")
def large_area_config_fixture(config_folder):
    """We use the "area": ... mapping so that it is of the same form as `config`."""
    filename = os.path.join(config_folder, "other", "large_area_global_config.json")
    return {"area": interpret_config_from_path(filename)}


def test_area_shape(storage, config):
    area_manager = UtmZoneAreaManager.from_raw_config(config["area"], storage)

    area_dataframe = area_manager.get_area_dataframe()

    assert isinstance(area_dataframe, GeoDataFrame)
    assert len(area_dataframe.index) == 3

    geometry = area_manager.get_area_geometry()
    assert isinstance(geometry, Geometry)


@pytest.mark.parametrize(
    "simplification_factor,point_count", [(0, 128), (0.00001, 64), (0.0001, 25), (0.001, 10), (0.1, 5)]
)
def test_area_shape_simplification(storage, config, simplification_factor, point_count):
    config["area"]["area_simplification_factor"] = simplification_factor
    area_manager = UtmZoneAreaManager.from_raw_config(config["area"], storage)

    geometry = area_manager.get_area_geometry()
    assert count_points(geometry.geometry) == point_count


@pytest.mark.parametrize(
    "full_config, expected_zone_num, expected_bbox_num",
    [
        (pytest.lazy_fixture("config"), 1, 2),
        (pytest.lazy_fixture("large_area_config"), 71, 368),
    ],
)
@pytest.mark.parametrize("add_bbox_column", [True, False])
def test_bbox_split(storage, full_config, expected_zone_num, expected_bbox_num, add_bbox_column):
    area_manager = UtmZoneAreaManager.from_raw_config(full_config["area"], storage)

    grid = area_manager.get_grid(add_bbox_column=add_bbox_column)
    expected_columns = ["index_n", "index_x", "index_y", "total_num", "geometry"]
    if add_bbox_column:
        expected_columns.append("BBOX")

    _check_area_grid(
        grid,
        expected_zone_num,
        expected_bbox_num,
        check_bboxes=add_bbox_column,
        expected_columns=expected_columns,
    )

    area_manager.cache_grid()  # test that caching goes through


def test_get_grid_caching(storage, config):
    """Tests that subsequent calls use the cached version of the grid. NOTE: implementation specific test!"""
    area_manager = UtmZoneAreaManager.from_raw_config(config["area"], storage)

    grid1 = area_manager.get_grid()
    # the second call should not create a new split
    with mock.patch.object(UtmZoneAreaManager, "_create_new_split") as creation_mock:
        grid2 = area_manager.get_grid()
        creation_mock.assert_not_called()

    for gdf1, gdf2 in zip(grid1, grid2):
        # could potentially fail if the loaded grid doesn't have the same order :/
        assert_geodataframe_equal(gdf1, gdf2, check_index_type=False, check_dtype=False)


def _check_area_grid(grid, expected_zone_num, expected_bbox_num, check_bboxes, expected_columns):
    assert isinstance(grid, list)
    assert len(grid) == expected_zone_num

    bbox_count = 0
    for subgrid in grid:
        assert isinstance(subgrid, GeoDataFrame)
        assert subgrid.columns.tolist() == expected_columns
        bbox_count += len(subgrid.index)

        if check_bboxes:
            assert "BBOX" in subgrid
            assert all(isinstance(item, BBox) for item in subgrid["BBOX"].values)

    assert bbox_count == expected_bbox_num
