"""
Tests for fs_utils module
"""
import datetime
from itertools import chain, combinations

import boto3
import numpy as np
import pytest
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from moto import mock_s3

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox

from eogrow.utils.filter import check_if_features_exist, get_patches_with_missing_features

pytestmark = pytest.mark.fast

BUCKET_NAME = "mocked-test-bucket"
PATCH_NAMES = [f"eopatch{i}" for i in range(5)]
REAL_FEATURES = [FeatureType.BBOX, FeatureType.TIMESTAMP, (FeatureType.DATA, "data"), (FeatureType.MASK, "mask")]
MISSING_FEATURES = [FeatureType.META_INFO, (FeatureType.DATA, "no_data"), (FeatureType.MASK_TIMELESS, "mask")]


def _prepare_fs(filesystem, eopatch):
    """Saves the eopatch under the predefined names, where every second one only contains the BBOX and MASK"""
    for i, name in enumerate(PATCH_NAMES):
        eopatch.save(name, filesystem=filesystem, features=[FeatureType.BBOX, FeatureType.MASK] if i % 2 else ...)


@pytest.fixture(name="eopatch", scope="session")
def eopatch_fixture():
    eopatch = EOPatch()
    eopatch.mask["mask"] = np.zeros((2, 3, 3, 2), dtype=np.int16)
    eopatch.data["data"] = np.zeros((2, 3, 3, 2), dtype=np.int16)
    eopatch.timestamp = [datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.datetime(2017, 1, 4, 10, 14, 5)]
    eopatch.bbox = BBox((1, 2, 3, 4), CRS.WGS84)
    eopatch.scalar["my scalar with spaces"] = np.array([[1, 2, 3], [1, 2, 3]])
    eopatch.scalar_timeless["my timeless scalar with spaces"] = np.array([1, 2, 3])
    return eopatch


@pytest.fixture(name="mock_s3fs", scope="session")
def mock_s3fs_fixture(eopatch):
    with mock_s3():
        s3resource = boto3.resource("s3", region_name="eu-central-1")
        s3resource.create_bucket(Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
        mock_s3fs = S3FS(BUCKET_NAME)
        _prepare_fs(mock_s3fs, eopatch)
        yield mock_s3fs


@pytest.fixture(name="temp_fs", scope="session")
def temp_fs_fixture(eopatch):
    temp_fs = TempFS()
    _prepare_fs(temp_fs, eopatch)
    yield temp_fs


@pytest.mark.parametrize(
    "test_features, expected_result",
    [(list(features), True) for features in chain(*(combinations(REAL_FEATURES, i) for i in range(4)))]
    + [([missing], False) for missing in MISSING_FEATURES]
    + [(REAL_FEATURES[:i] + [missing] + REAL_FEATURES[i:], False) for i, missing in enumerate(MISSING_FEATURES)],
)
def test_check_if_features_exist(mock_s3fs, temp_fs, test_features, expected_result):
    for filesystem in [mock_s3fs, temp_fs]:
        assert check_if_features_exist(filesystem, PATCH_NAMES[0], test_features) == expected_result


@pytest.mark.parametrize(
    "features, expected_num",
    [
        ([], 0),
        ([FeatureType.BBOX], 0),
        ([FeatureType.BBOX, (FeatureType.DATA, "data")], 2),
        ([(FeatureType.DATA, "no_data"), (FeatureType.DATA, "data")], 5),
    ],
)
def test_get_patches_with_missing_features(mock_s3fs, temp_fs, features, expected_num):
    for filesystem in [mock_s3fs, temp_fs]:
        assert len(get_patches_with_missing_features(filesystem, "/", PATCH_NAMES, features)) == expected_num
