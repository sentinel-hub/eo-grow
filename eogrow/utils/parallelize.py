"""
Utilities for parallelization of the code
"""
import concurrent.futures
from typing import Callable, Optional


def parallelize_with_threads(function: Callable, *params, workers: Optional[int] = None) -> list:
    """A simple parallelization helper function"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, *params))
