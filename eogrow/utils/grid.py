"""
Utilities for working with area grids
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List, Sequence, Tuple

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.geometry import GeometryCollection

from sentinelhub import BBox


@dataclass
class GridTransformation:
    """A dataclass holding information about a transformation from a group of source bounding boxes into a group of
    target bounding boxes via an enclosing bounding box.

    The dataclass also holds a source and a target dataframe which contain any additional information about bounding
    boxes that would be required to use at any later stage.
    """

    enclosing_bbox: BBox
    source_bboxes: Tuple[BBox, ...]
    target_bboxes: Tuple[BBox, ...]
    source_df: DataFrame = field(default_factory=DataFrame)
    target_df: DataFrame = field(default_factory=DataFrame)

    def __post_init__(self) -> None:
        if not self.target_bboxes:
            raise ValueError("There should be at least 1 target bounding box to make this transformation valid.")


def create_transformations(
    source_grid: List[GeoDataFrame], target_grid: List[GeoDataFrame], match_columns: List[str]
) -> List[GridTransformation]:
    """Given a source and a target dataframe it splits them into groups. Items in each group have to have the same
    values in `match_columns` columns. Then it creates a transformation object for each group."""
    transformations = []
    crs_map = {gdf.crs: gdf for gdf in source_grid}
    for target_gdf in target_grid:
        source_gdf = crs_map.get(target_gdf.crs)

        target_map = dict(iter(target_gdf.groupby(by=match_columns)))

        source_map: DefaultDict[tuple, DataFrame] = defaultdict(DataFrame)
        if source_gdf is not None:
            source_map = defaultdict(DataFrame, iter(source_gdf.groupby(by=match_columns)))

        for group_values, target_group in target_map.items():
            source_group = source_map[group_values]

            source_bboxes = source_group.BBOX.tolist() if len(source_group) else []
            target_bboxes = target_group.BBOX.tolist()
            enclosing_bbox = get_enclosing_bbox(source_bboxes + target_bboxes)

            transformation = GridTransformation(
                enclosing_bbox=enclosing_bbox,
                source_bboxes=tuple(source_bboxes),
                target_bboxes=tuple(target_bboxes),
                source_df=source_group,
                target_df=target_group,
            )
            transformations.append(transformation)

    return transformations


def get_grid_bbox(bbox: BBox, index: Tuple[int, int], split: Tuple[int, int]) -> BBox:
    """For a bounding box from a regular grid of bounding boxes, its position in the grid and the split of the grid it
    calculates the bounding box of the grid.

    :param bbox: A single bounding box from a regular grid. Bounding box shouldn't have any buffer.
    :param index: A pair of horizontal and vertical indices of a bounding box in a grid.
    :param split: A pair of numbers of splits in horizontal and vertical direction.
    :return: A total bounding box of the grid.
    """
    lower_left = np.array(bbox.lower_left)
    upper_right = np.array(bbox.upper_right)

    size = upper_right - lower_left
    grid_lower_left = lower_left - index * size
    grid_upper_right = lower_left + split * size - index * size

    return BBox((*grid_lower_left, *grid_upper_right), crs=bbox.crs)


def get_enclosing_bbox(bboxes: Sequence[BBox]) -> BBox:
    """A minimal bounding box that encloses all given bounding boxes"""
    unique_crs = {bbox.crs for bbox in bboxes}
    if len(unique_crs) != 1:
        raise ValueError(f"Given bounding boxes should have a single unique CRS, found {unique_crs}")

    collection = GeometryCollection([bbox.geometry for bbox in bboxes])
    return BBox(collection.bounds, crs=bboxes[0].crs)
