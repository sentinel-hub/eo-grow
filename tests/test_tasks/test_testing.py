import datetime as dt

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType
from eolearn.core.utils.common import is_discrete_type
from sentinelhub import CRS, BBox

from eogrow.tasks.testing import (
    DummyRasterFeatureTask,
    DummyTimestampFeatureTask,
    GenerateRasterFeatureTask,
    UniformDistribution,
)


@pytest.fixture()
def dummy_eopatch() -> EOPatch:
    return EOPatch(bbox=BBox((0, 0, 1, 1), CRS.POP_WEB))


@pytest.mark.parametrize(
    ("feature_type", "shape", "dtype", "min_value", "max_value"),
    [
        (FeatureType.DATA, (10, 20, 21, 5), np.float32, -3, -1),
        (FeatureType.DATA_TIMELESS, (20, 21, 1), int, -10, 5),
        (FeatureType.MASK, (10, 4, 6, 7), np.uint8, 3, 3),
        (FeatureType.LABEL, (10, 17), bool, 0, 1),
        (FeatureType.SCALAR_TIMELESS, (100,), float, 5, np.inf),
    ],
)
def test_dummy_raster_feature_task(dummy_eopatch, feature_type, shape, dtype, min_value, max_value):
    dummy_eopatch.timestamps = ["2011-08-12"] * 10
    feature = feature_type, "FEATURE"
    task = DummyRasterFeatureTask(feature, shape=shape, dtype=dtype, min_value=min_value, max_value=max_value)
    eopatch = task.execute(dummy_eopatch)

    assert isinstance(eopatch, EOPatch)

    assert feature in eopatch
    assert len(eopatch.get_features()) == 3

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

    task = DummyTimestampFeatureTask(time_interval=(start_time, end_time), num_timestamps=timestamp_num)
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


@pytest.mark.parametrize(
    ("feature_type", "shape", "dtype", "min_value", "max_value"),
    [
        (FeatureType.DATA, (10, 20, 21, 5), np.float32, -3, -1.5),
        (FeatureType.DATA_TIMELESS, (83, 69, 1), int, -2, 5),
        (FeatureType.MASK, (10, 4, 16, 7), np.int8, -3, 7),
        (FeatureType.LABEL, (10, 271), bool, 0, 1),
    ],
)
@pytest.mark.parametrize("seed", [1, 22, 9182])
def test_generate_raster_feature_task_uniform(dummy_eopatch, feature_type, shape, dtype, min_value, max_value, seed):
    dummy_eopatch.timestamps = ["2011-08-12"] * 10
    feature = feature_type, "FEATURE"
    configuration = UniformDistribution(min_value, max_value)

    task = GenerateRasterFeatureTask(feature, shape=shape, dtype=dtype, distribution=configuration)
    eopatch = task.execute(dummy_eopatch, seed=seed)

    assert feature in eopatch
    assert len(eopatch.get_features()) == 3

    data: np.ndarray = eopatch[feature]
    assert data.shape == shape
    assert data.dtype == dtype
    assert np.amin(data) >= min_value
    assert np.amax(data) <= max_value

    # check that the data is uniform 'enough'
    if is_discrete_type(data.dtype):
        # bins dont work well with few int values
        values, bin_nums = np.unique(data, return_counts=True)
        assert np.min(values) == min_value
        assert np.max(values) == max_value
        threshold = data.size / len(bin_nums) * 0.9
        assert all(bin_nums > threshold)
    else:
        bin_nums, _ = np.histogram(data, bins=5, range=(min_value, max_value))
        threshold = data.size / 5 * 0.9
        assert all(bin_nums > threshold)


# TODO: test that seeds return the same results, test that uniform is uniform, test that normal is normal
# TODO: add task for metainfo
