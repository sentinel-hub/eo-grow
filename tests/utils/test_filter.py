import datetime
from itertools import repeat

import boto3
import numpy as np
import pytest
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from moto import mock_s3

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox

from eogrow.utils.filter import check_if_features_exist, get_patches_with_missing_features

BUCKET_NAME = "mocked-test-bucket"
PATCH_NAMES = [f"eopatch{i}" for i in range(5)]
MISSING = [(FeatureType.META_INFO, "beep"), (FeatureType.DATA, "no_data"), (FeatureType.MASK_TIMELESS, "mask")]
EXISTING = [(FeatureType.DATA, "data"), (FeatureType.DATA, "data2"), (FeatureType.MASK, "mask")]


def _prepare_fs(filesystem, eopatch: EOPatch):
    """Saves the eopatch under the predefined names, where every second one only contains the BBOX and MASK"""
    for i, name in enumerate(PATCH_NAMES):
        eopatch.save(name, filesystem=filesystem, features=... if i % 2 else [FeatureType.MASK], save_timestamps=i > 1)


@pytest.fixture(name="eopatch", scope="session")
def eopatch_fixture():
    eopatch = EOPatch(bbox=BBox((1, 2, 3, 4), CRS.WGS84))
    eopatch.timestamps = [datetime.datetime(2017, 1, 1, 10, 4, 7), datetime.datetime(2017, 1, 4, 10, 14, 5)]
    eopatch.mask["mask"] = np.zeros((2, 3, 3, 2), dtype=np.int16)
    eopatch.data["data"] = np.zeros((2, 3, 3, 2), dtype=np.int16)
    eopatch.data["data2"] = np.zeros((2, 3, 3, 2), dtype=bool)
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
    return temp_fs


@pytest.mark.parametrize(
    ("test_features", "expected_result"),
    [
        ([EXISTING[0]], True),
        (EXISTING[1:], True),
        (EXISTING, True),
        *(([missing], False) for missing in MISSING),
        *((EXISTING[:i] + [missing] + EXISTING[i:], False) for i, missing in enumerate(MISSING)),
    ],
)
def test_check_if_features_exist(mock_s3fs, temp_fs, test_features, expected_result):
    for filesystem in [mock_s3fs, temp_fs]:
        # take the fourth patch because the first and third have missing features and the second has missing timestamps
        assert check_if_features_exist(filesystem, PATCH_NAMES[3], test_features) == expected_result


@pytest.mark.parametrize(
    ("features", "expected_num"),
    [
        ([], 2),  # timestamps are missing
        ([(FeatureType.DATA, "data")], 4),
        ([(FeatureType.DATA, "no_data"), (FeatureType.DATA, "data")], 5),
    ],
)
def test_get_patches_with_missing_features(mock_s3fs, temp_fs, features, expected_num):
    patch_list = list(zip(PATCH_NAMES, repeat(BBox((0, 0, 1, 1), CRS.WGS84))))
    for filesystem in [mock_s3fs, temp_fs]:
        assert len(get_patches_with_missing_features(filesystem, "/", patch_list, features)) == expected_num
