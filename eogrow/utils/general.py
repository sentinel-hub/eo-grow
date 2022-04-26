"""
A module containing general utilities that haven't been sorted in any other module
"""
import datetime as dt
import math
from enum import Enum
from typing import Tuple

import numpy as np
from aenum import MultiValueEnum

from sentinelhub import BBox, DataCollection
from sentinelhub.data_collections import DataCollectionDefinition


def jsonify(param: object) -> str:
    """Transforms an object into a normal string."""
    if isinstance(param, (dt.datetime, dt.date)):
        return param.isoformat()

    if isinstance(param, DataCollectionDefinition):
        return DataCollection(param).name
    if isinstance(param, DataCollection):
        return param.name

    if isinstance(param, MultiValueEnum):
        return param.values[0]

    if isinstance(param, Enum):
        return param.value

    raise TypeError(f"Object of type {type(param)} is not yet supported in jsonify utility function")


def reduce_to_coprime(number1: int, number2: int) -> Tuple[int, int]:
    """Divides given numbers by their greatest common divisor, thus making them coprime."""
    gcd = math.gcd(number1, number2)
    return number1 // gcd, number2 // gcd


def convert_to_int(values: np.ndarray, raise_diff: bool, error: float = 1e-8) -> np.ndarray:
    """Converts an array of floats into array of integers.

    :param values: An array of float values to be converted.
    :param raise_diff: Whether to raise an error if float values differ from integer values for more than the expected
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
    fixed_coords = convert_to_int(coords, raise_diff=True, error=error)
    return BBox(tuple(fixed_coords), crs=bbox.crs)
