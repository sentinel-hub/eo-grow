import datetime as dt

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType

from eogrow.tasks.testing import DummyRasterFeatureTask, DummyTimestampFeatureTask

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "feature_type, shape, dtype, min_value, max_value",
    [
        (FeatureType.DATA, (10, 20, 21, 5), np.float32, -3, -1),
        (FeatureType.DATA_TIMELESS, (20, 21, 1), int, -10, 5),
        (FeatureType.MASK, (4, 4, 6, 7), np.uint8, 3, 3),
        (FeatureType.LABEL, (10, 17), bool, 0, 1),
        (FeatureType.SCALAR_TIMELESS, (100,), float, 5, np.inf),
    ],
)
def test_dummy_raster_feature_task(feature_type, shape, dtype, min_value, max_value):
    feature = feature_type, "FEATURE"
    task = DummyRasterFeatureTask(feature, shape=shape, dtype=dtype, min_value=min_value, max_value=max_value)
    eopatch = task.execute()

    assert isinstance(eopatch, EOPatch)

    assert feature in eopatch
    assert len(eopatch.get_features()) == 1

    data = eopatch[feature]
    assert data.shape == shape
    assert data.dtype == dtype
    assert np.amin(data) >= min_value
    assert np.amax(data) <= max_value

    eopatch1 = task.execute(seed=42)
    eopatch2 = task.execute(seed=42)
    assert eopatch1 == eopatch2


def test_dummy_timestamp_feature_task():
    start_time = dt.datetime(year=2020, month=1, day=1)
    end_time = dt.date(year=2020, month=2, day=1)
    timestamp_num = 50

    task = DummyTimestampFeatureTask(time_interval=(start_time, end_time), timestamp_num=timestamp_num)
    eopatch = task.execute()

    assert isinstance(eopatch, EOPatch)

    assert FeatureType.TIMESTAMP in eopatch
    assert len(eopatch.get_features()) == 1

    assert len(eopatch.timestamp) == timestamp_num
    assert eopatch.timestamp == sorted(eopatch.timestamp)
    assert eopatch.timestamp[0] >= start_time
    assert eopatch.timestamp[-1] < dt.datetime.fromordinal(end_time.toordinal())

    eopatch1 = task.execute(seed=10)
    eopatch2 = task.execute(seed=10)
    assert eopatch1 == eopatch2
    assert eopatch1.timestamp[0].isoformat() == "2020-01-01T06:02:38"

    eopatch3 = task.execute(seed=11)
    assert eopatch1 != eopatch3
