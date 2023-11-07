"""Implements tasks needed for calculating features in FeaturesPipeline."""

from __future__ import annotations

import abc
from datetime import date, datetime, time, timedelta
from typing import Sequence

import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType, MapFeatureTask
from eolearn.core.types import Feature
from eolearn.core.utils.parsing import parse_renamed_feature


def join_valid_and_cloud_masks(valid_mask: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """Used to zip together information about valid data and clouds into a combined validity mask"""
    return valid_mask.astype(bool) & (cloud_mask == 0)


class ValidDataFractionPredicate:
    """
    Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, validity_threshold: float):
        self.validity_threshold = validity_threshold

    def __call__(self, array: np.ndarray) -> bool:
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return (coverage > self.validity_threshold).astype(bool)


class MaxNDVI(MapFeatureTask):
    def map_method(self, feature: np.ndarray) -> np.ndarray:
        if feature.shape[0]:
            return np.nanmax(feature, axis=0)
        # A special case of arrays with time dimension of size 0
        return np.full(feature.shape[1:], np.nan, dtype=feature.dtype)


class MosaickingTask(EOTask, metaclass=abc.ABCMeta):
    """Base class for mosaicking images given an interval of edge dates"""

    def __init__(
        self,
        feature: Feature,
        dates: list[date] | tuple[date, date, int],
        valid_mask: Feature | None = None,
        ndvi_feature: Feature | None = None,
    ):
        self.parsed_feature = parse_renamed_feature(feature, allowed_feature_types={FeatureType.DATA})
        self.valid_mask_type, self.valid_mask_name = None, None
        if valid_mask is not None:
            self.valid_mask_type, self.valid_mask_name = self.parse_feature(
                valid_mask, allowed_feature_types={FeatureType.MASK}
            )
        self.ndvi_feature_type, self.ndvi_feature_name = None, None
        if ndvi_feature is not None:
            self.ndvi_feature_type, self.ndvi_feature_name = self.parse_feature(
                ndvi_feature, allowed_feature_types={FeatureType.DATA}
            )
        self.dates = self._get_dates(dates)

    def _get_dates(self, dates: list[date] | tuple[date, date, int]) -> np.ndarray:
        """Set dates either from list of dates or a tuple (start_date, end_date, n_mosaics)"""
        if all(isinstance(d, (date, datetime)) for d in dates):
            return np.array(dates)
        if len(dates) == 3 and isinstance(dates[-1], int):
            return self._get_date_edges(*dates)
        raise ValueError(
            "dates parameter can be either a list of date(time)s or a tuple "
            "(start_date, end_date, n_mosaics) for equidistant intervals between start and end date."
        )

    @staticmethod
    def _get_date_edges(start_date: date, end_date: date, parts: int) -> np.ndarray:
        """Help function to get dates of year split into equal parts

        :param start_date: first date of time interval
        :param end_date: last date of time interval
        :param parts: Number of parts to split the year into
        :return: numpy array of dates that split the time interval into equal parts
        """
        start = datetime.combine(start_date, time.min)
        end = datetime.combine(end_date, time.min) + timedelta(days=1)
        diff = (end - start) / parts
        edges = [start + diff * i for i in range(parts)]
        edges.append(end)
        return np.array(edges)

    def _find_time_indices(self, timestamps: Sequence[date], index: int) -> tuple[np.ndarray, ...]:
        """Compute indices of images to use for mosaicking"""
        if index == 1:
            array = np.where(np.array(timestamps) <= self.dates[index])
        elif index == len(self.dates) - 1:
            array = np.where(np.array(timestamps) > self.dates[index - 1])
        else:
            array = np.where(
                (np.array(timestamps) > self.dates[index - 1]) & (np.array(timestamps) <= self.dates[index])
            )
        return array

    def compute_mosaic_dates(self) -> list[datetime]:
        """Compute dates of corresponding mosaics"""
        # calculate centers of date edges
        delta = self.dates[1:] - self.dates[:-1]
        return list(self.dates[:-1] + delta / 2)

    @abc.abstractmethod
    def _compute_single_mosaic(self, eopatch: EOPatch, idate: int) -> np.ndarray:
        """Compute single mosaic given index of edge date"""

    def compute_mosaic(self, eopatch: EOPatch) -> np.ndarray:
        """Computes mosaic"""
        return np.array([self._compute_single_mosaic(eopatch, idate) for idate in range(1, len(self.dates))])

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Compute mosaic for given dates"""
        feature_type, _, new_feature_name = self.parsed_feature
        output_patch = EOPatch(bbox=eopatch.bbox, timestamps=self.compute_mosaic_dates())

        eopatch.timestamps = [ts.replace(tzinfo=None) for ts in eopatch.get_timestamps()]
        output_patch[feature_type, new_feature_name] = self.compute_mosaic(eopatch)

        return output_patch


class MaxNDVIMosaickingTask(MosaickingTask):
    """
    Task to create mosaics of data based on the max NDVI value between provided dates
    """

    def __init__(
        self,
        feature: Feature,
        dates: list[date] | tuple[date, date, int],
        ndvi_feature: Feature,
        valid_mask: Feature | None = None,
    ):
        super().__init__(feature, dates, ndvi_feature=ndvi_feature, valid_mask=valid_mask)

    def _compute_single_mosaic(self, eopatch: EOPatch, idate: int) -> np.ndarray:
        """Compute single mosaic using values of the max NDVI"""
        array = self._find_time_indices(eopatch.get_timestamps(), idate)
        feature_type, feature_name, _ = self.parsed_feature
        feat_values = eopatch[(feature_type, feature_name)][array].astype(np.float32)
        ndvi_values = eopatch[(self.ndvi_feature_type, self.ndvi_feature_name)][array]  # type: ignore[index]
        valid_mask = (
            eopatch[self.valid_mask_type][self.valid_mask_name][array]
            if self.valid_mask_type is not None
            else np.ones(feat_values.shape, dtype=bool)
        ).astype(bool)

        ndvi_values[~valid_mask] = np.nan
        feat_values[~np.broadcast_to(valid_mask, feat_values.shape)] = np.nan

        mask_nan_slices = np.all(np.isnan(ndvi_values), axis=0, keepdims=True)
        ndvi_values[np.broadcast_to(mask_nan_slices, ndvi_values.shape)] = -999
        feat_values[np.broadcast_to(mask_nan_slices, feat_values.shape)] = -999

        timeframes, height, width, depth = feat_values.shape

        if timeframes == 0:
            mosaic = np.full((height, width, depth), np.nan)
        else:
            if timeframes == 1:
                mosaic = feat_values[0]
            else:
                indices = np.nanargmax(ndvi_values, axis=0).squeeze(axis=-1)
                ixgrid: tuple[np.ndarray, ...] = np.ix_(np.arange(timeframes), np.arange(height), np.arange(width))
                mosaic = feat_values[indices, ixgrid[1], ixgrid[2], :].squeeze(axis=0)

            mosaic[np.broadcast_to(mask_nan_slices[0], mosaic.shape)] = np.nan
        return mosaic


class MedianMosaickingTask(MosaickingTask):
    """
    Task to create mosaics of data based on the median value between provided dates
    """

    def __init__(
        self,
        feature: Feature,
        dates: list[date] | tuple[date, date, int],
        valid_mask: Feature | None = None,
    ):
        super().__init__(feature, dates, valid_mask=valid_mask)

    def _compute_single_mosaic(self, eopatch: EOPatch, idate: int) -> np.ndarray:
        """Compute single mosaic using the median of values"""
        array = self._find_time_indices(eopatch.get_timestamps(), idate)
        feature_type, feature_name, _ = self.parsed_feature

        feat_values = eopatch[(feature_type, feature_name)][array].astype(np.float32)
        valid_mask = (
            eopatch[(self.valid_mask_type, self.valid_mask_name)][array]
            if self.valid_mask_type is not None
            else np.ones(feat_values.shape, dtype=bool)
        )

        feat_values[~np.broadcast_to(valid_mask, feat_values.shape)] = np.nan

        mask_nan_slices = np.all(np.isnan(feat_values), axis=0, keepdims=True)
        feat_values[np.broadcast_to(mask_nan_slices, feat_values.shape)] = -999
        timeframes, height, width, depth = feat_values.shape

        if timeframes == 0:
            mosaic = np.full((height, width, depth), np.nan)
        else:
            if timeframes == 1:
                mosaic = feat_values[0]
            else:
                mosaic = np.nanmedian(feat_values, axis=0)

            mosaic[np.broadcast_to(mask_nan_slices[0], mosaic.shape)] = np.nan

        return mosaic
