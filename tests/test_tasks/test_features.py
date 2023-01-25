from typing import List, Tuple

import numpy as np
import pytest

from eogrow.tasks.features import ValidDataFractionPredicate

pytestmark = pytest.mark.fast

ZEROS = np.zeros((1, 100, 100))
ONES = np.ones((1, 100, 100))
MIX_CLEAN = np.concatenate([np.zeros((1, 100, 100)), np.ones((1, 100, 100))])
MIX_WIRD = np.concatenate([np.zeros((1, 100, 100)), np.ones((1, 100, 100))])
np.random.shuffle(MIX_WIRD)


@pytest.mark.parametrize(
    "threshold, test_cases",
    [
        (0.0, [(ZEROS, False), (MIX_CLEAN, True), (MIX_WIRD, True), (ONES, True)]),
        (0.49, [(ZEROS, False), (MIX_CLEAN, True), (MIX_WIRD, True), (ONES, True)]),
        (0.5, [(ZEROS, False), (MIX_CLEAN, False), (MIX_WIRD, False), (ONES, True)]),
        (1.0, [(ZEROS, False), (MIX_CLEAN, False), (MIX_WIRD, False), (ONES, False)]),
    ],
)
def test_ValidDataFractionPredicate(threshold: float, test_cases: List[Tuple[np.ndarray, bool]]) -> None:
    test_predicate = ValidDataFractionPredicate(threshold)
    for case in test_cases:
        assert test_predicate(case[0]) == case[1]
