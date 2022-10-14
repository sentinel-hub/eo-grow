"""
Tests for BYOC ingestion pipeline

This pipeline does not produce any data and is set to work on S3 only, so we need quite a bit of mocking to
achieve meaningful tests.
"""
import os
import time

import mock
import pytest
from shapely.geometry import Polygon

from sentinelhub import CRS
from sentinelhub.geometry import Geometry

from eogrow.core.storage import StorageManager
from eogrow.pipelines.byoc import IngestByocTilesPipeline
from eogrow.pipelines.export_maps import ExportMapsPipeline

CONFIG_SUBFOLDER = "byoc"
MOCK_COVER_GEOM = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]


@pytest.fixture(name="requests_mock")
def request_mock_setup(requests_mock):
    # logging
    requests_mock.get(url="/latest/dynamic/instance-identity/document", response_list=[{}])

    requests_mock.post(
        url="/oauth/token",
        response_list=[{"json": {"access_token": "x", "expires_in": 1000, "expires_at": time.time() + 10000}}],
    )

    # creating a new collection
    requests_mock.post(
        url="/api/v1/byoc/collections",
        response_list=[{"json": {"data": {"id": "mock-collection"}}}],
    )

    # searching for existing tiles
    requests_mock.get(
        url="/api/v1/byoc/collections/mock-collection/tiles",
        response_list=[{"json": {"data": [], "links": {}}}],
    )

    # creating new tiles
    requests_mock.post(
        url="/api/v1/byoc/collections/mock-collection/tiles",
        response_list=[{"json": {"data": {"id": i}}} for i in range(100)],
    )
    return requests_mock


def _get_tile_cover_geometry_mock(_: str) -> Geometry:
    return Geometry(Polygon(MOCK_COVER_GEOM), crs=CRS.WGS84)


@pytest.mark.parametrize(
    "preparation_config, config",
    [
        ("prepare_lulc_data.json", "ingest_lulc.json"),
    ],
)
@pytest.mark.order(after=["test_rasterize.py::test_rasterize_pipeline_features"])
def test_timeless_byoc(config_folder, preparation_config, config, requests_mock):
    preparation_config_path = os.path.join(config_folder, CONFIG_SUBFOLDER, preparation_config)
    ExportMapsPipeline.from_path(preparation_config_path).run()

    # patch storage manager so it'll believe it's on aws, but only during init
    with mock.patch.object(StorageManager, "is_on_aws", lambda _: True):
        config_path = os.path.join(config_folder, CONFIG_SUBFOLDER, config)
        pipeline = IngestByocTilesPipeline.from_path(config_path)
        pipeline.bucket_name = "mock-bucket"

    assert not pipeline.storage.is_on_aws()

    # Rasterio crashes when trying to open a file whose path is reconfigured to suit the bucket
    with mock.patch.object(pipeline, "_get_tile_cover_geometry", _get_tile_cover_geometry_mock):
        pipeline.run()

    # filter out all requests pertaining to logging of instance details
    relevant_requests = [req for req in requests_mock.request_history if "instance-identity" not in req.url]
    assert len(relevant_requests) == 4

    # first call is to authenticate
    auth_request = relevant_requests.pop(0)
    assert auth_request.url == "https://services.sentinel-hub.com/oauth/token"

    # create collection with correct name and mocked bucket name (because we're running locally)
    create_request = relevant_requests.pop(0)
    assert create_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections"
    assert create_request.method == "POST"
    assert create_request.json()["name"] == pipeline.config.new_collection_name
    assert create_request.json()["s3Bucket"] == "mock-bucket"

    # check existing tiles
    check_request = relevant_requests.pop(0)
    assert check_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections/mock-collection/tiles"
    assert check_request.method == "GET"

    for tile_request in relevant_requests:
        # create new tiles with correct path, sensing time, and cover geom
        assert tile_request.url == "https://services.sentinel-hub.com/api/v1/byoc/collections/mock-collection/tiles"
        assert tile_request.method == "POST"
        content = tile_request.json()
        assert content["coverGeometry"]["coordinates"] == [MOCK_COVER_GEOM]
        # it cannot strip the s3://<bucket-name> prefix since tests are local, so a full path is expected
        assert content["path"] == pipeline.config.storage.project_folder + "/maps/LULC_ID/UTM_32638"
        assert content["sensingTime"] == pipeline.config.sensing_time.isoformat() + "Z"
