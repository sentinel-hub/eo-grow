"""
A module containing general utilities that haven't been sorted in any other module
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

import numpy as np

from sentinelhub import BBox, DataCollection
from sentinelhub.data_collections import DataCollectionDefinition


def jsonify(param: object) -> str | list:
    """Transforms an object into a normal string."""
    if isinstance(param, set):
        return list(param)

    if isinstance(param, (dt.datetime, dt.date)):
        return param.isoformat()

    if isinstance(param, DataCollectionDefinition):
        return DataCollection(param).name
    if isinstance(param, DataCollection):
        return param.name

    if isinstance(param, Enum):
        return param.value

    raise TypeError(f"Object of type {type(param)} is not yet supported in jsonify utility function")


def convert_to_int(values: np.ndarray, raise_diff: bool, error: float = 1e-8) -> np.ndarray:
    """Converts an array of floats into array of integers.

    :param values: An array of float values to be converted.
    :param raise_diff: Raise an error if float values differ from integer values for more than the expected
        error.
    :param error: A joined maximal expected numerical error.
    """
    rounded_values = np.round(values)

    if raise_diff:
        diff = np.amax(np.absolute(values - rounded_values))
        if diff > error:
            raise ValueError(
                "Values can't be converted to integers because the difference to nearest integer values is larger "
                "than expected numerical error"
            )

    return rounded_values.astype(int)


def convert_bbox_coords_to_int(bbox: BBox, error: float = 1e-8) -> BBox:
    """Converts bounding box coordinates to integers by removing numerical errors. If the difference is larger than a
    numerical error it raises an error."""
    coords = np.array(list(bbox))
    min_x, min_y, max_x, max_y = convert_to_int(coords, raise_diff=True, error=error)
    return BBox((min_x, min_y, max_x, max_y), crs=bbox.crs)


def large_list_repr(large_list: list) -> str:
    """Creates a representation of a large list of elements that consists only of a representation of first 3 and the
    last element."""
    if len(large_list) <= 4:
        return repr(large_list)

    first_elements = ", ".join(map(repr, large_list[:3]))
    return f"[{first_elements}, ..., {large_list[-1]}]"


def current_timestamp(fmt: str = "%Y-%m-%dT%H-%M-%SZ") -> str:
    """Creates a timestamp string of the current time"""
    return dt.datetime.utcnow().strftime(fmt)
