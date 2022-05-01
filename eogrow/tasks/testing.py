"""
Implements tasks for testing purposes
"""
import datetime as dt
from typing import Optional, Tuple

import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureTypeSet

from ..utils.types import Feature, TimePeriod


class DummyRasterFeatureTask(EOTask):
    """Creates a raster feature with random values according to"""

    def __init__(self, feature: Feature, shape: Tuple[int, ...], dtype: np.dtype, min_value=0, max_value=1):
        """
        :param feature: A raster feature to be created.
        :param shape: Shape of the feature
        :param dtype:
        :param min_value:
        :param max_value:
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=FeatureTypeSet.RASTER_TYPES)
        self.shape = shape
        self.dtype = dtype
        self.min_value = min_value
        self.max_value = max_value

        feature_type, _ = self.feature
        if len(self.shape) != feature_type.ndim():
            raise ValueError(
                f"Feature {self.feature} should have {feature_type.ndim()}-dimensional shape but {self.shape} was given"
            )

    def _get_random_raster(self, rng: np.random.Generator) -> np.ndarray:
        """Creates a raster array from given random generator."""
        if self.max_value == self.min_value:
            return np.full(self.shape, self.max_value, dtype=self.dtype)

        array = rng.random(size=self.shape)
        array = (self.max_value - self.min_value) * array + self.min_value
        if self.dtype is not None:
            array = array.astype(self.dtype)
        return array

    def execute(self, eopatch: Optional[EOPatch] = None, seed: Optional[int] = None) -> EOPatch:
        """Generates a raster feature randomly with a given seed."""
        eopatch = eopatch or EOPatch()
        rng = np.random.default_rng(seed)

        eopatch[self.feature] = self._get_random_raster(rng)
        return eopatch


class DummyTimestampFeatureTask(EOTask):
    """Adds random timestamps according to given parameters"""

    def __init__(self, time_interval: TimePeriod, timestamp_num: int):
        """
        :param time_interval: A time interval `[start, end)` from where all timestamps will be generated.
        :param timestamp_num: Number of timestamp in the created timestamp feature.
        """
        self.time_interval = tuple(map(_ensure_datetime, time_interval))
        self.timestamp_num = timestamp_num

    def execute(self, eopatch: Optional[EOPatch] = None, seed: Optional[int] = None) -> EOPatch:
        """Generates timestamps randomly with a given seed."""
        eopatch = eopatch or EOPatch()
        rng = np.random.default_rng(seed)

        start_time, end_time = self.time_interval
        total_seconds = (end_time - start_time).total_seconds()
        random_integers = rng.integers(total_seconds, size=self.timestamp_num)
        random_integers.sort()
        timestamps = [start_time + dt.timedelta(seconds=int(seconds)) for seconds in random_integers]

        eopatch.timestamp = timestamps
        return eopatch


def _ensure_datetime(timestamp: dt.date) -> dt.datetime:
    """Ensures that the given timestamp is a datetime and not a date object."""
    if isinstance(timestamp, dt.datetime):
        return timestamp
    return dt.datetime.fromordinal(timestamp.toordinal())
