"""
Tasks for spatial operations on EOPatches
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from geopandas import GeoDataFrame

from eolearn.core import EOPatch, EOTask, FeatureType, deep_eq
from sentinelhub import CRS, BBox, bbox_to_resolution

from ..utils.general import convert_to_int
from ..utils.types import Feature, FeatureSpec
from ..utils.vector import concat_gdf


class SpatialJoinTask(EOTask):
    """Spatially joins data from multiple EOPatches"""

    ERR = 1e-8

    def __init__(
        self,
        features: List[FeatureSpec],
        no_data_map: Dict[Feature, float],
        unique_columns_map: Dict[Feature, List[str]],
        raise_misaligned: bool = True,
    ):
        self.features = self.parse_features(features)
        self.no_data_map = no_data_map
        self.unique_columns_map = unique_columns_map
        self.raise_misaligned = raise_misaligned

    def _join_spatial_rasters(
        self, rasters: List[np.ndarray], bboxes: List[BBox], joined_bbox: BBox, no_data_value: float
    ) -> np.ndarray:
        """Joins all rasters into a single new rasters."""
        if len(set(raster.dtype for raster in rasters)) != 1:
            raise ValueError("Cannot join raster features with different dtypes")

        resolution = self._get_resolution(rasters, bboxes)
        joined_raster = self._get_joined_raster(joined_bbox, resolution, rasters[0], no_data_value)
        height, width = joined_raster.shape[-3:-1]

        for raster, bbox in zip(rasters, bboxes):
            slice_x, slice_y = get_array_slices(
                joined_bbox,
                bbox,
                resolution=resolution,
                raise_misaligned=self.raise_misaligned,
                limit_x=(0, width),
                limit_y=(0, height),
            )
            joined_raster[..., slice_y, slice_x, :] = raster

        return joined_raster

    def _get_resolution(self, rasters: List[np.ndarray], bboxes: List[BBox]) -> np.ndarray:
        """Checks that resolutions of given geospatial rasters don't differ more than a rounding error."""
        resolutions = []

        for raster, bbox in zip(rasters, bboxes):
            height, width = raster.shape[-3:-1]
            resolutions.append(np.array(bbox_to_resolution(bbox, width=width, height=height, meters=False)))

        mean_resolution = np.mean(resolutions, axis=0)
        for resolution in resolutions:
            diff = np.amax(np.absolute(resolution - mean_resolution))
            if diff > self.ERR:
                raise ValueError("Rasters have different spatial resolutions therefore they cannot be joined together")

        return mean_resolution

    def _get_joined_raster(
        self, bbox: BBox, resolution: np.ndarray, sample_raster: np.ndarray, no_data_value: float
    ) -> np.ndarray:
        """Provides an empty raster for a given bbox and resolution."""
        image_size = np.abs(np.array(bbox.upper_right) - np.array(bbox.lower_left)) / resolution
        try:
            width, height = convert_to_int(image_size, raise_diff=self.raise_misaligned)
        except ValueError as exception:
            raise ValueError(
                "Pixels from joined raster would be misaligned with pixels from original rasters"
            ) from exception

        time_dim = sample_raster.shape[:-3]  # Either empty or with a single element
        shape = (*time_dim, height, width, sample_raster.shape[-1])
        return np.full(shape, no_data_value, dtype=sample_raster.dtype)

    @staticmethod
    def _join_vector_data(dataframes: List[GeoDataFrame], unique_columns: Optional[List[str]]) -> GeoDataFrame:
        """Joins dataframes and optionally drops duplicated rows."""
        joined_dataframe = concat_gdf(dataframes)
        if unique_columns:
            return joined_dataframe.drop_duplicates(subset=unique_columns)
        return joined_dataframe

    def execute(self, *eopatches: EOPatch, bbox: BBox) -> EOPatch:
        """Spatially joins given EOPatches into a new EOPatch with given bounding box."""
        eopatches = tuple(eopatch for eopatch in eopatches if eopatch.get_features())
        if not all(eopatch.bbox for eopatch in eopatches):
            raise ValueError("All non-empty input EOPatches should have a bounding box")
        if any(eopatch.bbox.crs is not bbox.crs for eopatch in eopatches):
            raise ValueError("EOPatches must have the same CRS as the given bounding box")

        # Sorting EOPatches in unique order so that the values in the overlapping areas will always be computed in
        # the same way.
        eopatches = tuple(sorted(eopatches, key=lambda eop: repr(eop.bbox)))

        joined_eopatch = EOPatch(bbox=bbox)

        for feature in self.features:
            feature_type, _ = feature
            if feature_type is FeatureType.BBOX:
                continue

            data = [eopatch[feature] for eopatch in eopatches if feature in eopatch]
            if not data:
                continue

            if feature_type.is_spatial():
                if feature_type.is_raster():
                    bboxes = [eopatch.bbox for eopatch in eopatches if feature in eopatch]
                    joined_data = self._join_spatial_rasters(data, bboxes, bbox, self.no_data_map[feature])
                else:
                    joined_data = self._join_vector_data(data, self.unique_columns_map.get(feature))
            else:
                joined_data = data[0]

                if not all(deep_eq(joined_data, item) for item in data[1:]):
                    raise ValueError(
                        f"Different EOPatches have different values of a non-spatial feature {feature}. It is not clear"
                        " how to join them."
                    )

            joined_eopatch[feature] = joined_data

        return joined_eopatch


class SpatialSliceTask(EOTask):
    """Spatially slices given EOPatch to create a new one."""

    def __init__(self, features: List[FeatureSpec], raise_misaligned: bool = True):
        self.features = self.parse_features(features)
        self.raise_misaligned = raise_misaligned

    def _slice_raster(self, raster: np.ndarray, raster_bbox: BBox, slice_bbox: BBox) -> np.ndarray:
        """Spatially slice a raster array."""
        height, width = raster.shape[-3:-1]
        slice_x, slice_y = get_array_slices(
            raster_bbox, slice_bbox, size=(width, height), raise_misaligned=self.raise_misaligned
        )
        return raster[..., slice_y, slice_x, :]

    @staticmethod
    def _filter_vector(gdf: GeoDataFrame, bbox: BBox) -> GeoDataFrame:
        """Spatially filters a GeoDataFrame."""
        bbox = bbox.transform(CRS(gdf.crs))
        intersects_bbox = gdf.geometry.intersects(bbox.geometry)
        return gdf[intersects_bbox].copy(deep=True)

    def execute(self, eopatch: EOPatch, *, bbox: Optional[BBox]) -> EOPatch:
        """Spatially slices given EOPatch with given bounding box.

        The task allows that no bounding box for slicing is provided. In such case the task will return an EOPatch with
        only non-spatial features.
        """
        main_bbox = eopatch.bbox
        if not main_bbox:
            raise ValueError("EOPatch is missing a bounding box")
        if bbox is not None:
            if main_bbox.crs is not bbox.crs:
                raise ValueError("Given bbox is not in the same CRS as EOPatch bbox")
            if not main_bbox.geometry.contains(bbox.geometry):
                raise ValueError("Given bbox must be fully contained in EOPatch's bbox")

        sliced_eopatch = EOPatch(bbox=bbox)

        for feature in self.features:
            feature_type, _ = feature
            if feature_type is FeatureType.BBOX:
                continue
            data = eopatch[feature]

            if feature_type.is_spatial():
                if bbox is None:
                    continue

                if feature_type.is_raster():
                    sliced_data = self._slice_raster(data, main_bbox, bbox)
                else:
                    sliced_data = self._filter_vector(data, bbox)
            else:
                sliced_data = data

            sliced_eopatch[feature] = sliced_data

        return sliced_eopatch


def get_array_slices(
    bbox: BBox,
    slice_bbox: BBox,
    *,
    resolution: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
    size: Optional[Union[np.ndarray, Tuple[int, int]]] = None,
    raise_misaligned: bool = True,
    limit_x: Optional[Tuple[int, int]] = None,
    limit_y: Optional[Tuple[int, int]] = None,
) -> Tuple[slice, slice]:
    """Slicing taken from eolearn.io.ImportFromTiffTask.

    :param bbox: A bounding box of initial array.
    :param slice_bbox: A bounding box of array to be sliced.
    :param resolution: A working resolution.
    :param size: A working size.
    :param raise_misaligned: Whether to raise an error if the slice would be pixel misaligned the initial array.
    :param limit_x: If provided it will clip the horizontal slice to a given interval.
    :param limit_y: If provided it will clip the vertical slice to a given interval.
    :return: A slice over horizontal direction and a slice over vertical direction.
    """
    raster_upper_left = np.array([bbox.min_x, bbox.max_y])

    if size is not None and resolution is None:
        width, height = size
        raster_lower_right = np.array([bbox.max_x, bbox.min_y])
        resolution = np.abs((raster_upper_left - raster_lower_right)) / (width, height)
    elif size is None and resolution is not None:
        resolution = np.array(resolution)
    else:
        raise ValueError("Only one of the resolution and size can be given")

    slice_upper_left = np.array([slice_bbox.min_x, slice_bbox.max_y])
    slice_lower_right = np.array([slice_bbox.max_x, slice_bbox.min_y])
    axis_flip = [1, -1]  # image origin is upper left, geographic origin is lower left

    offset = axis_flip * (slice_upper_left - raster_upper_left) / resolution
    try:
        offset_x, offset_y = convert_to_int(offset, raise_diff=raise_misaligned)
        slice_size = abs(slice_lower_right - slice_upper_left) / resolution
        slice_width, slice_height = convert_to_int(slice_size, raise_diff=raise_misaligned)
    except ValueError as exception:
        raise ValueError(
            "Pixels from a slice raster would be misaligned with pixels from an original raster"
        ) from exception

    interval_x = offset_x, offset_x + slice_width
    if limit_x:
        interval_x = max(interval_x[0], limit_x[0]), min(interval_x[1], limit_x[1])
    interval_y = offset_y, offset_y + slice_height
    if limit_y:
        interval_y = max(interval_y[0], limit_y[0]), min(interval_y[1], limit_y[1])

    return slice(*interval_x), slice(*interval_y)
