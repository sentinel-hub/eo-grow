import os
import time

import pytest
from geopandas import GeoDataFrame

from sentinelhub import BBox, Geometry

from eogrow.core.area import UtmZoneAreaManager
from eogrow.core.config import interpret_config_from_path
from eogrow.utils.vector import count_points

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="large_area_config")
def large_area_config_fixture(config_folder):
    filename = os.path.join(config_folder, "other", "large_area_global_config.json")
    return interpret_config_from_path(filename)


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


# No idea how to use @pytest.mark.parametrize over config, large_area_config
def test_bbox_split(storage, config, large_area_config):
    for area_config, expected_zone_num, expected_bbox_num in [
        (config["area"], 1, 2),
        (large_area_config, 71, 368),
    ]:
        area_manager = UtmZoneAreaManager.from_raw_config(area_config, storage)

        start_time = time.perf_counter()
        grid = area_manager.get_grid(add_bbox_column=True)
        splitting_time = time.perf_counter() - start_time

        _check_area_grid(
            grid,
            expected_zone_num,
            expected_bbox_num,
            check_bboxes=True,
            expected_columns=["index_n", "index_x", "index_y", "total_num", "geometry", "BBOX"],
        )

        start_time = time.perf_counter()
        grid = area_manager.get_grid()
        assert time.perf_counter() - start_time < max(splitting_time / 2, 1)  # Checking if data is kept in the class

        _check_area_grid(
            grid,
            expected_zone_num,
            expected_bbox_num,
            check_bboxes=False,
            expected_columns=["index_n", "index_x", "index_y", "total_num", "geometry"],
        )

        area_manager.cache_grid()


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
