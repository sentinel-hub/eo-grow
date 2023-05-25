from typing import List, Tuple

import numpy as np
import pytest

from eogrow.tasks.features import ValidDataFractionPredicate

ZEROES = np.zeros((1, 100, 100))
ONES = np.ones((1, 100, 100))
NOT_SYMMETRICAL = np.array([x % 3 == 0 for x in range(1000)]).reshape(10, 10, 10)
MIX = np.concatenate([np.zeros((10000)), np.ones((10000))])
np.random.shuffle(MIX)
MIX.reshape(2, 100, 100)
MIX_BOOL = MIX.astype(bool)


@pytest.mark.parametrize(
    "threshold, test_cases",
    [
        (0.0, [(ZEROES, False), (NOT_SYMMETRICAL, True), (MIX, True), (MIX_BOOL, True), (ONES, True)]),
        (0.49, [(ZEROES, False), (NOT_SYMMETRICAL, False), (MIX, True), (MIX_BOOL, True), (ONES, True)]),
        (0.5, [(ZEROES, False), (NOT_SYMMETRICAL, False), (MIX, False), (MIX_BOOL, False), (ONES, True)]),
        (1.0, [(ZEROES, False), (NOT_SYMMETRICAL, False), (MIX, False), (MIX_BOOL, False), (ONES, False)]),
    ],
)
def test_valid_data_fraction_predicate(threshold: float, test_cases: List[Tuple[np.ndarray, bool]]) -> None:
    test_predicate = ValidDataFractionPredicate(threshold)
    for case in test_cases:
        assert test_predicate(case[0]) == case[1]
