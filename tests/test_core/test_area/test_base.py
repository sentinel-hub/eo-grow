from typing import Dict, List, Literal, Tuple, cast
from unittest.mock import patch

import fs
import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from sentinelhub import CRS, BBox
from sentinelhub.geometry import Geometry

from eogrow.core.area.base import BaseAreaManager, get_geometry_from_file
from eogrow.core.config import RawConfig
from eogrow.core.storage import StorageManager
from eogrow.utils.eopatch_list import save_names
from eogrow.utils.vector import count_points


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


def test_get_grid_filtration(storage: StorageManager) -> None:
    config = _prepare_patch_list_config(storage, ["beep"])
    filtered_grid = DummyAreaManager.from_raw_config(config, storage).get_grid()

    assert len(filtered_grid) == 1

    expected_geoms = GeoDataFrame(
        data={"eopatch_name": ["beep"]}, geometry=[BBox((0, 0, 1, 1), CRS.WGS84).geometry], crs=CRS.WGS84.pyproj_crs()
    )
    assert_geodataframe_equal(filtered_grid[CRS.WGS84], expected_geoms)


def test_get_grid_filtration_flag(storage: StorageManager) -> None:
    full_grid = DummyAreaManager.from_raw_config({}, storage).get_grid()
    config = _prepare_patch_list_config(storage, ["beep"])
    unfiltered_grid = DummyAreaManager.from_raw_config(config, storage).get_grid(filtered=False)

    for crs, crs_grid in full_grid.items():
        assert_geodataframe_equal(unfiltered_grid[crs], crs_grid)


def test_get_grid_filtration_failure(storage: StorageManager) -> None:
    config = _prepare_patch_list_config(storage, ["I_do_not_exist"])
    with pytest.raises(ValueError):
        DummyAreaManager.from_raw_config(config, storage).get_grid()


def test_get_grid_caching(storage: StorageManager) -> None:
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
    ("patch_list", "expected_bboxes"),
    [
        ([], []),
        (None, list(zip(DummyAreaManager.NAMES, DummyAreaManager.BBOXES))),
        (DummyAreaManager.NAMES[1:], list(zip(DummyAreaManager.NAMES[1:], DummyAreaManager.BBOXES[1:]))),
    ],
)
def test_get_patch_list(patch_list: List[str], expected_bboxes: List[Tuple[str, BBox]], storage: StorageManager):
    config = {} if patch_list is None else _prepare_patch_list_config(storage, patch_list)
    manager = DummyAreaManager.from_raw_config(config, storage)

    assert expected_bboxes == manager.get_patch_list()

    assert expected_bboxes == manager.get_patch_list(), "Filtration fails on reading from cache."


@pytest.mark.parametrize("engine", ["fiona", "pyogrio"])
def test_get_geometry_from_file(storage: StorageManager, engine: Literal["fiona", "pyogrio"]):
    file_path = fs.path.join(storage.get_input_data_folder(), "test_area.geojson")

    geometry = get_geometry_from_file(storage.filesystem, file_path, engine)
    assert count_points(geometry.geometry) == 20


def _prepare_patch_list_config(storage: StorageManager, patch_list: List[str]) -> RawConfig:
    path = fs.path.join(storage.get_folder("temp"), "patch_list.json")
    save_names(storage.filesystem, path, patch_list)
    config = {"patch_names": {"input_folder_key": "temp", "filename": "patch_list.json"}}
    return cast(RawConfig, config)
