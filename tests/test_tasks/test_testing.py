import datetime as dt

import pytest

from eolearn.core import EOPatch, FeatureType

from eogrow.tasks.testing import DummyTimestampFeatureTask

pytestmark = pytest.mark.fast


def test_dummy_timestamp_feature_task():
    start_time = dt.datetime(year=2020, month=1, day=1)
    end_time = dt.date(year=2020, month=2, day=1)
    timestamp_num = 50

    task = DummyTimestampFeatureTask(time_interval=(start_time, end_time), timestamp_num=timestamp_num)
    eopatch = task.execute()

    assert isinstance(eopatch, EOPatch)
    assert eopatch.get_feature_list() == [FeatureType.TIMESTAMP]
    assert len(eopatch.timestamp) == timestamp_num
    assert eopatch.timestamp == sorted(eopatch.timestamp)
    assert eopatch.timestamp[0] >= start_time
    assert eopatch.timestamp[-1] < dt.datetime.fromordinal(end_time.toordinal())

    old_timestamps = eopatch.timestamp
    eopatch = task.execute(eopatch)
    assert eopatch.timestamp != old_timestamps

    eopatch1 = task.execute(seed=10)
    eopatch2 = task.execute(seed=10)
    assert eopatch1 == eopatch2
    assert eopatch1.timestamp[0].isoformat() == "2020-01-01T06:02:38"
