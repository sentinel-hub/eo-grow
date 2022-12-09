"""
Tests for BatchAreaManager

Since it relies heavily on the SH Batch API we have to mock a lot.

Mocks:
- Batch request definition endpoint.
- Tiling grid request endpoints.
- Mocking requests of iter_tiles would be too much effort, so the `_make_new_split` of the splitter is mocked instead.
"""
from unittest.mock import patch

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from sentinelhub import CRS, BBox
from sentinelhub.areas import BatchSplitter

from eogrow.core.area import NewBatchAreaManager
from eogrow.core.area.batch import MissingBatchIdError

pytestmark = pytest.mark.fast


@pytest.fixture(name="configured_requests_mock")
def request_mock_setup(requests_mock):
    requests_mock.post(url="/oauth/token", real_http=True)

    batch_bounds = {
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[[[-17.5773, 14.77391], [-17.5752, 14.6874], [-17.4505, 14.7845], [-17.5773, 14.7739]]]],
        },
        "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
    }
    batch_def = {
        "id": "fake-id",
        "tileCount": 3,
        "status": "DONE",
        "tilingGrid": {"id": 2, "resolution": 120.0, "bufferX": 10, "bufferY": 1},
        "tileWidthPx": 100,
        "tileHeightPx": 100,
        "processRequest": {"input": {"bounds": batch_bounds}},
    }
    requests_mock.get(
        url="https://services.sentinel-hub.com/api/v1/batch/process/fake-id", response_list=[{"json": batch_def}]
    )

    requests_mock.get(url="https://services.sentinel-hub.com/api/v1/batch/tilinggrids/2", real_http=True)

    return requests_mock


@pytest.fixture(scope="function", name="area_config")
def area_config_fixture():
    return {
        "manager": "eogrow.core.area.NewBatchAreaManager",
        "area": {
            "filename": "test_large_area.geojson",
            "buffer": 2,
            "simplification_factor": 0.1,
        },
        "tiling_grid_id": 2,
        "resolution": 120,
        "tile_buffer_x": 10,
        "tile_buffer_y": 1,
        "batch_id": "fake-id",
    }


def test_cache_name(storage, area_config):
    manager = NewBatchAreaManager.from_raw_config(area_config, storage)

    assert manager.get_grid_cache_filename() == "NewBatchAreaManager_test_large_area_2_120.0_10_1.gpkg"


def test_no_batch_id_error(storage, area_config):
    del area_config["batch_id"]
    manager = NewBatchAreaManager.from_raw_config(area_config, storage)
    with pytest.raises(MissingBatchIdError):
        manager.get_grid()


def test_grid(storage, area_config, configured_requests_mock):
    manager = NewBatchAreaManager.from_raw_config(area_config, storage)

    bboxes = [BBox((0, 0, 1, 1), CRS.WGS84), BBox((1, 1, 2, 2), CRS.WGS84), BBox((0, 0, 1, 1), CRS(3035))]
    tiles = (bboxes, [{"name": name} for name in ["beep", "boop", "bap"]])
    with patch.object(BatchSplitter, "_make_split", lambda *_: tiles):
        grid = manager.get_grid()

    expected_grid = {
        CRS.WGS84: GeoDataFrame(
            data={"eopatch_name": ["beep", "boop"]},
            geometry=[bbox.geometry for bbox in bboxes[:2]],
            crs=CRS.WGS84.pyproj_crs(),
        ),
        CRS(3035): GeoDataFrame(
            data={"eopatch_name": ["bap"]},
            geometry=[bboxes[2].geometry],
            crs=CRS(3035).pyproj_crs(),
        ),
    }
    for crs in (CRS.WGS84, CRS(3035)):
        assert_geodataframe_equal(grid[crs], expected_grid[crs], check_index_type=False, check_dtype=False)
