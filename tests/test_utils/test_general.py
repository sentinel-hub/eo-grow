import datetime as dt
import enum
import json

import numpy as np
import pytest

from eolearn.core import FeatureType
from sentinelhub import CRS, BBox, DataCollection

from eogrow.utils.general import convert_bbox_coords_to_int, convert_to_int, jsonify, large_list_repr, reduce_to_coprime


class MyEnum(enum.Enum):
    NO_DATA = "no data"


ORIGINAL_CONFIG = {
    "feature_types": (FeatureType.DATA, FeatureType.BBOX),
    "enum": MyEnum.NO_DATA,
    "timestamp": dt.datetime(year=2021, month=9, day=30),
    "collection_set": {DataCollection.SENTINEL2_L1C},
    "collection_def": DataCollection.SENTINEL2_L1C.value,
}
SERIALIZED_CONFIG = {
    "feature_types": ["data", "bbox"],
    "enum": "no data",
    "timestamp": "2021-09-30T00:00:00",
    "collection_set": ["SENTINEL2_L1C"],
    "collection_def": "SENTINEL2_L1C",
}


def test_jsonify():
    new_config = json.loads(json.dumps(ORIGINAL_CONFIG, default=jsonify))
    assert new_config == SERIALIZED_CONFIG

    with pytest.raises(TypeError):
        json.dumps({"some function": json.dumps}, default=jsonify)


@pytest.mark.parametrize(
    "number1, number2, expected_result",
    [
        (2, 3, (2, 3)),
        (48, 32, (3, 2)),
        (1, 1, (1, 1)),
        (10**20, 10**21, (1, 10)),
    ],
)
def test_reduce_to_coprime(number1, number2, expected_result):
    result = reduce_to_coprime(number1, number2)
    assert result == expected_result


def test_convert_to_int():
    array = np.array([0, 0.999999999, 1e-8], dtype=float)
    rounded_array = np.array([0, 1, 0], dtype=float)

    assert np.array_equal(convert_to_int(array, raise_diff=True), rounded_array)
    assert np.array_equal(convert_to_int(array, raise_diff=False), rounded_array)

    with pytest.raises(ValueError):
        assert np.array_equal(convert_to_int(array, raise_diff=True, error=1e-9), rounded_array)
    assert np.array_equal(convert_to_int(array, raise_diff=False, error=1e-9), rounded_array)


def test_convert_bbox_coords_to_int():
    bbox = BBox([1.00000001, -2, 3, -1e-8], CRS.WGS84)
    rounded_bbox = BBox([1, -2, 3, 0], CRS.WGS84)

    assert convert_bbox_coords_to_int(bbox) == rounded_bbox
    with pytest.raises(ValueError):
        assert convert_bbox_coords_to_int(bbox, error=1e-9)


@pytest.mark.parametrize(
    "values_num, expected_repr",
    [
        (0, "[]"),
        (3, "[0, 1, 2]"),
        (4, "[0, 1, 2, 3]"),
        (10, "[0, 1, 2, ..., 9]"),
    ],
)
def test_large_list_repr(values_num, expected_repr):
    values = list(range(values_num))
    values_repr = large_list_repr(values)
    assert values_repr == expected_repr
