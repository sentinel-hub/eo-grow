"""A collection of functions to be used with the ZipMapPipeline."""

from typing import Dict, Optional, Union

import numpy as np


def map_values(
    array: np.ndarray,
    *,
    mapping: Dict[int, int],
    default: Optional[int] = None,
    dtype: Union[None, np.dtype, type] = None
) -> np.ndarray:
    """Maps all values from `array` according to a dictionary. A default value can be given, which is assigned to
    values without a corresponding key in the mapping dictionary.

    Vectorizing the `.get` method is faster, but difficult to do with a default (and also loses speed advantage)."""
    if default is not None:
        return np.array([mapping.get(x, default) for x in array.ravel()], dtype=dtype).reshape(array.shape)
    return np.vectorize(mapping.get)(array).astype(dtype)
