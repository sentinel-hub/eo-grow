"""
Implements tasks for testing purposes
"""
import datetime as dt
from typing import Optional, Tuple, List

import numpy as np

from eolearn.core import EOTask, EOPatch, FeatureType

from ..utils.types import Feature, TimePeriod


class DummyRasterFeatureTask(EOTask):
    def __init__(self, feature: Feature, shape: Optional[Tuple[int, ...]] = None, dtype=None, min_value=0, max_value=1):
        self.feature = self.parse_feature(feature)
        self.shape = shape
        self.dtype = dtype
        self.min_value = min_value
        self.max_value = max_value

        feature_type = self.feature[0]
        if not feature_type.is_raster():  # TODO: mandatory for rasters, but also for others?
            if self.shape is not None:
                raise ValueError("Shape can only be given for raster features")
            if self.dtype is not None:
                raise ValueError("Dtype can only be given for raster features")

    def _get_random_raster(self, rng: np.random.Generator) -> np.ndarray:
        if self.max_value == self.min_value:
            return np.full(self.shape, self.max_value, dtype=self.dtype)

        array = rng.random(size=self.shape)
        array = (self.max_value - self.min_value) * array + self.min_value
        if self.dtype is not None:
            array = array.astype(self.dtype)
        return array

    def _get_random_timestamps(self, rng: np.random.Generator) -> List[dt.datetime]:
        pass

    def execute(self, eopatch: Optional[EOPatch] = None, seed: Optional[int] = None) -> EOPatch:
        eopatch = eopatch or EOPatch()
        rng = np.random.default_rng(seed)

        eopatch[self.feature] = self._get_random_raster(rng)
        return eopatch


class DummyTimestampFeatureTask(EOTask):
    def __init__(self, time_interval: TimePeriod, timestamp_num: int):
        self.time_interval = time_interval
        self.timestamp_num = timestamp_num

    def execute(self, eopatch: Optional[EOPatch] = None, seed: Optional[int] = None) -> EOPatch:
        eopatch = eopatch or EOPatch()
        rng = np.random.default_rng(seed)

        start_time, end_time = self.time_interval
        total_seconds = (end_time - start_time).total_seconds()
        random_integers = rng.integers(total_seconds, size=self.timestamp_num)
        random_integers.sort()
        timestamps = [start_time + dt.timedelta(seconds=int(seconds)) for seconds in random_integers]

        eopatch.timestamp = timestamps
        return eopatch
