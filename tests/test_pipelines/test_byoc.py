"""
Tests for BYOC ingestion pipeline

This pipeline does not produce any data and is set to work on S3 only, so we need quite a bit of mocking to
achieve meaningful tests.

Mocks:
- The pipeline has to be tricked into thinking it's on AWS during init so it doesn't raise an error.
- The bucket_name is changed since it's not parsed correctly (because we're not on S3).
- When reading the cover geometry the path is not filesystem relative (but a join of bucket name + bucket relative)
    so we mock it to prevent `rasterio.open` to fail.
- Most request endpoints are mocked in the `requests_mock` fixture.
"""
from unittest.mock import patch

import pytest
from shapely.geometry import Polygon

from sentinelhub import CRS, SentinelHubDownloadClient
from sentinelhub.geometry import Geometry

from eogrow.core.storage import StorageManager
from eogrow.pipelines.byoc import IngestByocTilesPipeline
from eogrow.utils.testing import run_config

CONFIG_SUBFOLDER = "byoc"
MOCK_COVER_GEOM = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]

pytestmark = pytest.mark.integration


@pytest.fixture(name="configured_requests_mock")
def request_mock_setup(requests_mock):
    requests_mock.get(url="/latest/dynamic/instance-identity/document", real_http=True)  # logging
    requests_mock.post(url="/oauth/token", real_http=True)

    # creating a new collection
    requests_mock.post(url="/api/v1/byoc/collections", response_list=[{"json": {"data": {"id": "mock-collection"}}}])

    requests_mock.get(  # searching for existing tiles
        url="/api/v1/byoc/collections/mock-collection/tiles", response_list=[{"json": {"data": [], "links": {}}}]
    )

    requests_mock.post(  # creating new tiles
        url="/api/v1/byoc/collections/mock-collection/tiles", response_list=[{"json": {"data": {"id": 0}}}]
    )
    return requests_mock


def run_byoc_pipeline(config_path: str, requests_mock):
    # mock tile cover geom
    def _get_tile_cover_geometry_mock(_: str) -> Geometry:
        return Geometry(Polygon(MOCK_COVER_GEOM), crs=CRS.WGS84)

    # patch storage manager so it believes it's on aws, but only during init
    with patch.object(StorageManager, "is_on_s3", lambda _: True):
        SentinelHubDownloadClient._CACHED_SESSIONS = {}
        pipeline = IngestByocTilesPipeline.from_path(config_path)
        pipeline.bucket_name = "mock-bucket"

    with patch.object(pipeline, "_get_tile_cover_geometry", _get_tile_cover_geometry_mock):
        pipeline.run()

    # filter out all requests pertaining to logging of instance details
    relevant_requests = [req for req in requests_mock.request_history if "instance-identity" not in req.url]

    return pipeline, relevant_requests


@pytest.mark.chain
@pytest.mark.parametrize("preparation_config, config", [("prepare_lulc_data", "ingest_lulc")])
@pytest.mark.order(after=["test_rasterize.py::test_rasterize_pipeline_features"])
def test_timeless_byoc(config_and_stats_paths, preparation_config, config, configured_requests_mock):
    preparation_config_path, _ = config_and_stats_paths("byoc", preparation_config)
    config_path, _ = config_and_stats_paths("byoc", config)

    run_config(preparation_config_path)
    pipeline, requests = run_byoc_pipeline(config_path, configured_requests_mock)

    auth_request = requests.pop(0)
    assert auth_request.url == "https://services.sentinel-hub.com/oauth/token"

    creation_request = requests.pop(0)
    assert creation_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections"
    assert creation_request.method == "POST"
    assert creation_request.json()["name"] == pipeline.config.new_collection_name
    assert creation_request.json()["s3Bucket"] == "mock-bucket"

    check_request = requests.pop(0)
    assert check_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections/mock-collection/tiles"
    assert check_request.method == "GET"

    for tile_request in requests:
        assert tile_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections/mock-collection/tiles"
        assert tile_request.method == "POST"

        content = tile_request.json()
        assert content["coverGeometry"]["coordinates"] == [MOCK_COVER_GEOM]
        assert content["path"] == pipeline.config.storage.project_folder + "/maps/LULC_ID/UTM_32638"
        assert content["sensingTime"] == pipeline.config.sensing_time.isoformat() + "Z"


@pytest.mark.parametrize("preparation_config, config", [("prepare_bands_data", "ingest_bands")])
@pytest.mark.order(after=["test_rasterize.py::test_rasterize_pipeline_features"])
def test_temporal_byoc(config_and_stats_paths, preparation_config, config, configured_requests_mock):
    preparation_config_path, _ = config_and_stats_paths("byoc", preparation_config)
    config_path, _ = config_and_stats_paths("byoc", config)

    run_config(preparation_config_path)
    pipeline, requests = run_byoc_pipeline(config_path, configured_requests_mock)

    auth_request = requests.pop(0)
    assert auth_request.url == "https://services.sentinel-hub.com/oauth/token"

    creation_request = requests.pop(0)
    assert creation_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections"
    assert creation_request.method == "POST"
    assert creation_request.json()["name"] == pipeline.config.new_collection_name
    assert creation_request.json()["s3Bucket"] == "mock-bucket"

    check_request = requests.pop(0)
    assert check_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections/mock-collection/tiles"
    assert check_request.method == "GET"

    timestamps = [
        "2019-01-04T07:48:37Z",
        "2019-01-24T07:48:39Z",
        "2019-02-13T07:48:39Z",
        "2019-02-18T07:48:36Z",
        "2019-02-23T07:49:38Z",
        "2019-03-05T07:55:53Z",
    ]
    for tile_request in requests:
        assert tile_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections/mock-collection/tiles"
        assert tile_request.method == "POST"

        content = tile_request.json()
        assert content["coverGeometry"]["coordinates"] == [MOCK_COVER_GEOM]

        timestamp = content["sensingTime"]
        assert timestamp in timestamps
        timestamps.remove(timestamp)

        subfolder = timestamp.replace(":", "-").replace("Z", "")
        assert content["path"] == pipeline.config.storage.project_folder + f"/maps/BANDS-S2-L1C/UTM_32638/{subfolder}"

    assert not timestamps
