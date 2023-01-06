from typing import Dict
from unittest.mock import patch

import fs
import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from sentinelhub import CRS, BBox
from sentinelhub.geometry import Geometry

from eogrow.core.area.base import BaseAreaManager, get_geometry_from_file
from eogrow.utils.eopatch_list import save_eopatch_names
from eogrow.utils.vector import count_points

pytestmark = pytest.mark.fast


class DummyAreaManager(BaseAreaManager):
    BBOXES = [BBox((0, 0, 1, 1), CRS.WGS84), BBox((1, 1, 2, 2), CRS.WGS84), BBox((0, 0, 1, 1), CRS(3035))]
    NAMES = ["beep", "boop", "bap"]

    def _create_grid(self) -> Dict[CRS, GeoDataFrame]:
        return {
            CRS.WGS84: GeoDataFrame(
                data={"eopatch_name": self.NAMES[:2]},
                geometry=[bbox.geometry for bbox in self.BBOXES[:2]],
                crs=CRS.WGS84.pyproj_crs(),
            ),
            CRS(3035): GeoDataFrame(
                data={"eopatch_name": [self.NAMES[2]]}, geometry=[self.BBOXES[2].geometry], crs=CRS(3035).pyproj_crs()
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


@pytest.mark.parametrize(
    "patch_list, expected_bboxes",
    [
        ([], []),
        (None, list(zip(DummyAreaManager.NAMES, DummyAreaManager.BBOXES))),
        (DummyAreaManager.NAMES[1:], list(zip(DummyAreaManager.NAMES[1:], DummyAreaManager.BBOXES[1:]))),
    ],
)
def test_get_names_and_bboxes(patch_list, expected_bboxes, storage):
    if patch_list is None:
        config = {}
    else:
        path = fs.path.join(storage.get_folder("temp"), "patch_list.json")
        save_eopatch_names(storage.filesystem, path, patch_list)
        config = {"patch_list": {"input_folder_key": "temp", "filename": "patch_list.json"}}

    manager = DummyAreaManager.from_raw_config(config, storage)

    assert expected_bboxes == manager.get_patch_list()


@pytest.mark.parametrize(
    "simplification_factor,expected_point_count", [(0, 128), (0.00001, 64), (0.0001, 25), (0.001, 10), (0.1, 5)]
)
@pytest.mark.parametrize("engine", ["fiona", "pyogrio"])
def test_get_geometry_from_file(storage, simplification_factor, expected_point_count, engine):
    file_path = fs.path.join(storage.get_input_data_folder(), "test_area.geojson")

    geometry = get_geometry_from_file(storage.filesystem, file_path, 0.001, simplification_factor, engine)
    assert count_points(geometry.geometry) == expected_point_count
