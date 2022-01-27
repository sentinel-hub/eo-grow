"""
Tests for general utilities
"""
import json
import datetime as dt

import pytest
from sentinelhub import DataCollection
from eolearn.core import FeatureType

from eogrow.utils.enum import BaseEOGrowEnum
from eogrow.utils.general import jsonify


class MyEnum(BaseEOGrowEnum):
    NO_DATA = "no data", 0, "#ffffff"


ORIGINAL_CONFIG = {
    "feature_types": (FeatureType.DATA, FeatureType.BBOX),
    "multi_value_enum": MyEnum.NO_DATA,
    "timestamp": dt.datetime(year=2021, month=9, day=30),
    "collection": DataCollection.SENTINEL2_L1C,
    "collection_def": DataCollection.SENTINEL2_L1C.value,
}
SERIALIZED_CONFIG = {
    "feature_types": ["data", "bbox"],
    "multi_value_enum": "no data",
    "timestamp": "2021-09-30T00:00:00",
    "collection": "SENTINEL2_L1C",
    "collection_def": "SENTINEL2_L1C",
}


@pytest.mark.fast
def test_jsonify():
    new_config = json.loads(json.dumps(ORIGINAL_CONFIG, default=jsonify))
    assert new_config == SERIALIZED_CONFIG

    with pytest.raises(TypeError):
        json.dumps({"some function": json.dumps}, default=jsonify)
