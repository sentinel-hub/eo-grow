"""Tasks used to generate test data."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import Feature
from eolearn.core.utils.common import is_discrete_type

from ..types import TimePeriod


@dataclass
class UniformDistribution:
    min_value: float
    max_value: float


@dataclass
class NormalDistribution:
    mean: float
    std: float


class GenerateRasterFeatureTask(EOTask):
    """Creates a raster feature with random values"""

    def __init__(
        self,
        feature: Feature,
        shape: tuple[int, ...],
        dtype: np.dtype | type,
        distribution: UniformDistribution | NormalDistribution,
    ):
        """
        :param feature: A raster feature to be created.
        :param shape: Shape of the created feature array.
        :param dtype: A dtype of the feature.
        :param distribution: The distribution for generating values.
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=lambda fty: fty.is_array())
        self.shape = shape
        self.dtype = dtype
        self.distribution = distribution

        feature_type, _ = self.feature
        if len(self.shape) != feature_type.ndim():
            raise ValueError(
                f"Feature {self.feature} should have {feature_type.ndim()}-dimensional shape but {self.shape} was given"
            )
        if feature_type.is_discrete() and not is_discrete_type(self.dtype):
            raise ValueError(f"Feature {self.feature} only supports discrete dtypes but {self.dtype} was given")

    def _generate_data(
        self, configuration: NormalDistribution | UniformDistribution, rng: np.random.Generator
    ) -> np.ndarray:
        if isinstance(configuration, NormalDistribution):
            return rng.normal(configuration.mean, configuration.std, size=self.shape)

        if is_discrete_type(self.dtype):
            min_val, max_val = round(configuration.min_value), round(configuration.max_value)
            return rng.integers(min_val, max_val, size=self.shape, endpoint=True)
        array = rng.random(size=self.shape)
        return (configuration.max_value - configuration.min_value) * array + configuration.min_value

    def execute(self, eopatch: EOPatch, seed: int) -> EOPatch:
        """Generates a raster feature randomly with a given seed."""
        rng = np.random.default_rng(seed)

        generated_data = self._generate_data(self.distribution, rng)

        if is_discrete_type(self.dtype):
            generated_data = np.rint(generated_data)

        eopatch[self.feature] = generated_data.astype(self.dtype)
        return eopatch


class GenerateTimestampsTask(EOTask):
    """Creates a timestamp feature with random timestamps"""

    def __init__(self, time_interval: TimePeriod, num_timestamps: int):
        """
        :param time_interval: A time interval `[start, end)` from where all timestamps will be generated.
        :param timestamp_num: Number of timestamp in the created timestamp feature.
        """
        self.time_interval = tuple(map(_ensure_datetime, time_interval))
        self.num_timestamps = num_timestamps

    def execute(self, eopatch: EOPatch, seed: int) -> EOPatch:
        """Generates timestamps randomly with a given seed."""
        rng = np.random.default_rng(seed)

        start_time, end_time = self.time_interval
        total_seconds = int((end_time - start_time).total_seconds())
        random_integers = rng.integers(total_seconds, size=self.num_timestamps)
        random_integers.sort()
        timestamps = [start_time + dt.timedelta(seconds=int(seconds)) for seconds in random_integers]

        eopatch.timestamps = timestamps
        return eopatch


def _ensure_datetime(timestamp: dt.date) -> dt.datetime:
    """Ensures that the given timestamp is a datetime and not a date object."""
    if isinstance(timestamp, dt.datetime):
        return timestamp
    return dt.datetime.fromordinal(timestamp.toordinal())
