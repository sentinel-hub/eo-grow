"""Tasks used to generate test data."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from eolearn.core import EOPatch, EOTask
from eolearn.core.utils.common import is_discrete_type

from ..types import Feature, TimePeriod


@dataclass
class UniformDistribution:
    min_value: float
    max_value: float


@dataclass
class NormalDistribution:
    mean: float
    std: float


class DummyRasterFeatureTask(EOTask):
    """Creates a raster feature with random values"""

    def __init__(
        self,
        feature: Feature,
        shape: tuple[int, ...],
        dtype: np.dtype | type,
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
        self.feature = self.parse_feature(feature, allowed_feature_types=lambda fty: fty.is_array())
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

    def execute(self, eopatch: EOPatch, seed: int | None = None) -> EOPatch:
        """Generates a raster feature randomly with a given seed."""
        rng = np.random.default_rng(seed)

        eopatch[self.feature] = self._get_random_raster(rng)
        return eopatch


class GenerateRasterFeatureTask(EOTask):
    """Creates a raster feature with random values"""

    def __init__(
        self,
        feature: Feature,
        shape: tuple[int, ...],
        dtype: np.dtype | type,
        configuration: UniformDistribution | NormalDistribution,
    ):
        """
        :param feature: A raster feature to be created.
        :param shape: Shape of the created feature array.
        :param dtype: A dtype of the feature.
        :param configuration: The configuration for generating values.
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=lambda fty: fty.is_array())
        self.shape = shape
        self.dtype = dtype
        self.configuration = configuration

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
        if isinstance(configuration, UniformDistribution):
            array = rng.random(size=self.shape)
            return (configuration.max_value - configuration.min_value) * array + configuration.min_value
        return rng.normal(configuration.mean, configuration.std, size=self.shape)

    def execute(self, eopatch: EOPatch, seed: int | None = None) -> EOPatch:
        """Generates a raster feature randomly with a given seed."""
        rng = np.random.default_rng(seed)

        generated_data = self._generate_data(self.configuration, rng)

        if is_discrete_type(self.dtype):
            generated_data = np.rint(generated_data, dtype=self.dtype)
        else:
            generated_data = generated_data.astype(self.dtype)

        eopatch[self.feature] = generated_data
        return eopatch


class DummyTimestampFeatureTask(EOTask):
    """Creates a timestamp feature with random timestamps"""

    def __init__(self, time_interval: TimePeriod, num_timestamps: int):
        """
        :param time_interval: A time interval `[start, end)` from where all timestamps will be generated.
        :param timestamp_num: Number of timestamp in the created timestamp feature.
        """
        self.time_interval = tuple(map(_ensure_datetime, time_interval))
        self.num_timestamps = num_timestamps

    def execute(self, eopatch: EOPatch, seed: int | None = None) -> EOPatch:
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
