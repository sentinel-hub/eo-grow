"""
Modules with Ray-related utilities
"""
import logging
import os

import ray

from ..types import BoolOrAuto
from .general import current_timestamp

LOGGER = logging.getLogger(__name__)
CLUSTER_CONFIG_DIR = "~/.synced_configs"


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


def generate_cluster_config_path(config_filename: str) -> str:
    """Generate the path to the rsynced config on the cluster"""
    return f"{CLUSTER_CONFIG_DIR}/{current_timestamp()}{os.path.basename(config_filename)}"
