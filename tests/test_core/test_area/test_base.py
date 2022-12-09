from typing import Dict
from unittest.mock import patch

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from sentinelhub import CRS, BBox
from sentinelhub.geometry import Geometry

from eogrow.core.area.base import BaseAreaManager, BaseSplitterAreaManager
from eogrow.utils.vector import count_points

pytestmark = pytest.mark.fast


class DummyAreaManager(BaseAreaManager):
    def _create_grid(self) -> Dict[CRS, GeoDataFrame]:
        bboxes = [BBox((0, 0, 1, 1), CRS.WGS84), BBox((1, 1, 2, 2), CRS.WGS84), BBox((0, 0, 1, 1), CRS(3035))]
        return {
            CRS.WGS84: GeoDataFrame(
                data={"eopatch_name": ["beep", "boop"]},
                geometry=[bbox.geometry for bbox in bboxes[:2]],
                crs=CRS.WGS84.pyproj_crs(),
            ),
            CRS(3035): GeoDataFrame(
                data={"eopatch_name": ["bap"]}, geometry=[bboxes[2].geometry], crs=CRS(3035).pyproj_crs()
            ),
        }

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84) -> Geometry:
        raise NotImplementedError

    def get_grid_cache_filename(self) -> str:
        return "test.gpkg"


def test_get_grid_caching(storage):
    """Tests that subsequent calls use the cached version of the grid. NOTE: implementation specific test!"""
    manager = DummyAreaManager.from_raw_config({}, storage)

    grid1 = manager.get_grid()
    # the second call should not create a new split
    with patch.object(DummyAreaManager, "_create_grid") as creation_mock:
        grid2 = manager.get_grid()
        creation_mock.assert_not_called()

    for gdf1, gdf2 in zip(grid1.values(), grid2.values()):
        # could potentially fail if the loaded grid doesn't have the same order :/
        assert_geodataframe_equal(gdf1, gdf2, check_index_type=False, check_dtype=False)


class DummySplitterAreaManager(BaseSplitterAreaManager):
    def _create_grid(self) -> Dict[CRS, GeoDataFrame]:
        raise NotImplementedError

    def get_grid_cache_filename(self) -> str:
        raise NotImplementedError


@pytest.mark.parametrize(
    "simplification_factor,expected_point_count", [(0, 128), (0.00001, 64), (0.0001, 25), (0.001, 10), (0.1, 5)]
)
def test_base_splitter_area_geometry(storage, simplification_factor, expected_point_count):
    area_config = {
        "area": {"filename": "test_area.geojson", "buffer": 0.001, "simplification_factor": simplification_factor},
    }
    area_manager = DummySplitterAreaManager.from_raw_config(area_config, storage)

    geometry = area_manager.get_area_geometry()
    assert count_points(geometry.geometry) == expected_point_count
