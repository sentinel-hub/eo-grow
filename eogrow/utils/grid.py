"""
Utilities for working with area grids
"""

from __future__ import annotations

import numpy as np

from sentinelhub import BBox


def split_bbox(
    named_bbox: tuple[str, BBox],
    split_x: int,
    split_y: int,
    buffer_x: float,
    buffer_y: float,
    naming_schema: str = "{name}_{i_x}_{i_y}",
) -> list[tuple[str, BBox]]:
    """Splits a BBox into multiple smaller BBoxes with new names generated for them.

    The `buffer` parameters describe the buffer of the original BBox, which is copied to the split ones.
    """
    name, bbox = named_bbox
    min_x, min_y = bbox.lower_left
    max_x, max_y = bbox.upper_right

    x_edges, x_step = np.linspace(min_x + buffer_x, max_x - buffer_x, split_x, endpoint=False, retstep=True)
    y_edges, y_step = np.linspace(min_y + buffer_y, max_y - buffer_y, split_y, endpoint=False, retstep=True)

    split_bboxes = []
    for i_x, x_edge in enumerate(x_edges):
        for i_y, y_edge in enumerate(y_edges):
            lower_left = (x_edge - buffer_x, y_edge - buffer_y)
            upper_right = (x_edge + x_step + buffer_x, y_edge + y_step + buffer_y)
            split_bbox = BBox((lower_left, upper_right), crs=bbox.crs)

            split_name = naming_schema.format(name=name, i_x=i_x, i_y=i_y)
            split_bboxes.append((split_name, split_bbox))

    return split_bboxes
