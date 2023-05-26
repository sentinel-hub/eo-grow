import datetime as dt

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox

from eogrow.tasks.testing import DummyRasterFeatureTask, DummyTimestampFeatureTask


@pytest.fixture()
def dummy_eopatch() -> EOPatch:
    return EOPatch(bbox=BBox((0, 0, 1, 1), CRS.POP_WEB))


@pytest.mark.parametrize(
    ("feature_type", "shape", "dtype", "min_value", "max_value"),
    [
        (FeatureType.DATA, (10, 20, 21, 5), np.float32, -3, -1),
        (FeatureType.DATA_TIMELESS, (20, 21, 1), int, -10, 5),
        (FeatureType.MASK, (4, 4, 6, 7), np.uint8, 3, 3),
        (FeatureType.LABEL, (10, 17), bool, 0, 1),
        (FeatureType.SCALAR_TIMELESS, (100,), float, 5, np.inf),
    ],
)
def test_dummy_raster_feature_task(dummy_eopatch, feature_type, shape, dtype, min_value, max_value):
    feature = feature_type, "FEATURE"
    task = DummyRasterFeatureTask(feature, shape=shape, dtype=dtype, min_value=min_value, max_value=max_value)
    eopatch = task.execute(dummy_eopatch)

    assert isinstance(eopatch, EOPatch)

    assert feature in eopatch
    assert len(eopatch.get_features()) == 2

    data = eopatch[feature]
    assert data.shape == shape
    assert data.dtype == dtype
    assert np.amin(data) >= min_value
    assert np.amax(data) <= max_value

    eopatch1 = task.execute(dummy_eopatch, seed=42)
    eopatch2 = task.execute(dummy_eopatch, seed=42)
    assert eopatch1 == eopatch2


def test_dummy_timestamp_feature_task(dummy_eopatch: EOPatch):
    start_time = dt.datetime(year=2020, month=1, day=1)
    end_time = dt.date(year=2020, month=2, day=1)
    timestamp_num = 50

    task = DummyTimestampFeatureTask(time_interval=(start_time, end_time), timestamp_num=timestamp_num)
    eopatch = task.execute(dummy_eopatch)

    assert isinstance(eopatch, EOPatch)

    assert FeatureType.TIMESTAMPS in eopatch
    assert len(eopatch.get_features()) == 2

    assert len(eopatch.timestamps) == timestamp_num
    assert eopatch.timestamps == sorted(eopatch.timestamps)
    assert eopatch.timestamps[0] >= start_time
    assert eopatch.timestamps[-1] < dt.datetime.fromordinal(end_time.toordinal())

    eopatch1 = task.execute(dummy_eopatch.copy(), seed=10)
    eopatch2 = task.execute(dummy_eopatch.copy(), seed=10)
    assert eopatch1 == eopatch2
    assert eopatch1.timestamps[0].isoformat() == "2020-01-01T06:02:38"

    eopatch3 = task.execute(dummy_eopatch.copy(), seed=11)
    assert eopatch1 != eopatch3
