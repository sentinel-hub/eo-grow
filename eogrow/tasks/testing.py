"""
Implements tasks for testing purposes
"""
import datetime as dt
from typing import Optional, Tuple, Union

import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureTypeSet
from eolearn.core.utils.common import is_discrete_type

from ..utils.types import Feature, TimePeriod


class DummyRasterFeatureTask(EOTask):
    """Creates a raster feature with random values"""

    def __init__(
        self,
        feature: Feature,
        shape: Tuple[int, ...],
        dtype: Union[np.dtype, type],
        min_value: float = 0,
        max_value: float = 1,
    ):
        """
        :param feature: A raster feature to be created.
        :param shape: Shape of the created feature array.
        :param dtype: A dtype of the feature.
        :param min_value: All feature values will be greater or equal to this value.
        :param max_value: If feature has a discrete dtype or max_value == min_value then all feature values will be
            lesser or equal to this value. Otherwise, all features will be strictly lesser to this value.
        """
        self.feature: Feature = self.parse_feature(feature, allowed_feature_types=FeatureTypeSet.RASTER_TYPES)
        self.shape = shape
        self.dtype = dtype
        self.min_value = min_value
        self.max_value = max_value

        feature_type, _ = self.feature
        if len(self.shape) != feature_type.ndim():
            raise ValueError(
                f"Feature {self.feature} should have {feature_type.ndim()}-dimensional shape but {self.shape} was given"
            )
        if feature_type.is_discrete() and not is_discrete_type(self.dtype):
            raise ValueError(f"Feature {self.feature} only supports discrete dtypes but {self.dtype} was given")

    def _get_random_raster(self, rng: np.random.Generator) -> np.ndarray:
        """Creates a raster array from given random generator."""
        if self.max_value == self.min_value:
            return np.full(self.shape, self.max_value, dtype=self.dtype)

        if is_discrete_type(self.dtype):
            return rng.integers(
                int(self.min_value), int(self.max_value), size=self.shape, dtype=self.dtype, endpoint=True
            )

        array = rng.random(size=self.shape)
        array = (self.max_value - self.min_value) * array + self.min_value
        return array.astype(self.dtype)

    def execute(self, eopatch: Optional[EOPatch] = None, seed: Optional[int] = None) -> EOPatch:
        """Generates a raster feature randomly with a given seed."""
        eopatch = eopatch or EOPatch()
        rng = np.random.default_rng(seed)

        eopatch[self.feature] = self._get_random_raster(rng)
        return eopatch


class DummyTimestampFeatureTask(EOTask):
    """Creates a timestamp feature with random timestamps"""

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
        total_seconds = int((end_time - start_time).total_seconds())
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
