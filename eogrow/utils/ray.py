"""
Modules with Ray-related utilities
"""
import logging
import os

import ray

from ..types import BoolOrAuto
from .time import get_timestamp_suffix

LOGGER = logging.getLogger(__name__)


def handle_ray_connection(use_ray: BoolOrAuto = "auto") -> bool:
    """According to the given parameter it will try to connect to an existing Ray cluster.

    :param use_ray: Either a boolean flag or `"auto"` to define if the connection should be established or not.
    :return: `True` if connection is established and `False` otherwise.
    """
    if use_ray == "auto":
        try:
            _try_connect_to_ray()
            return True
        except ConnectionError:
            LOGGER.info("No Ray cluster found, will not use Ray.")
            return False

    if use_ray:
        _try_connect_to_ray()
        return True
    return False


def _try_connect_to_ray() -> None:
    """Try connecting and log if successful."""
    ray.init(address="auto", ignore_reinit_error=True)
    LOGGER.info("Connected to an existing Ray cluster.")


def get_cluster_config_path(config_filename: str) -> str:
    timestamp = get_timestamp_suffix()
    base, ext = os.path.splitext(os.path.basename(config_filename))
    return f"~/.synced_configs/{base}_{timestamp}{ext}"
