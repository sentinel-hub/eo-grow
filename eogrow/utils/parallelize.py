"""
Utilities for parallelization of the code
"""
from typing import Optional, Callable
import concurrent.futures


def parallelize_with_threads(function: Callable, *params, workers: Optional[int] = None) -> list:
    """A simple parallelization helper function"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, *params))
