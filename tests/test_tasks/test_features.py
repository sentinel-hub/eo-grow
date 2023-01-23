import numpy as np
import pytest
from numpy.testing import assert_array_equal

from eolearn.core import EOPatch
from eolearn.core.constants import FeatureType

from eogrow.tasks.features import MaxNDVI, ValidDataFractionPredicate


@pytest.mark.parametrize(
    "validity_threshold, data, expected",
    [
        (0.0, np.array([]), False),
        (0.9, np.array([True], dtype=bool), True),
        (0.0, np.zeros((3, 3, 2)), False),
        (0.45, np.concatenate([np.zeros(100), np.ones(100)]).reshape(2, 10, 10), True),
        (0.5, np.concatenate([np.zeros(100), np.ones(100)]).reshape(2, 10, 10), False),
    ],
)
def test_ValidDataFractionPredicate(validity_threshold: float, data: np.ndarray, expected: bool) -> None:
    test_predicate = ValidDataFractionPredicate(validity_threshold)
    assert test_predicate(data) == expected


@pytest.mark.parametrize(
    "ndvi, expected",
    [
        (np.ones((1, 1, 1, 1)), np.ones((1, 1, 1))),
        (np.ones((5, 10, 10, 1)), np.ones((10, 10, 1))),
        (np.arange(500).reshape((5, 10, 10, 1)), np.arange(400, 500).reshape((10, 10, 1))),
    ],
)
def test_MaxNDVI(ndvi: np.ndarray, expected: np.ndarray) -> None:
    patch = EOPatch()
    patch.data["NDVI"] = ndvi

    test_max_ndvi = MaxNDVI((FeatureType.DATA, "NDVI"), (FeatureType.DATA_TIMELESS, "MaxNDVI"))
    assert_array_equal(test_max_ndvi(patch).data_timeless["MaxNDVI"], expected)
