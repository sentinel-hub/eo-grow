"""
Modules with Ray-related utilities
"""

from __future__ import annotations

import logging
import os
import subprocess

from .general import current_timestamp

LOGGER = logging.getLogger(__name__)
CLUSTER_CONFIG_DIR = "~/.synced_configs"


def is_cluster_running(cluster_yaml: str) -> bool:
    """Checks if cluster is running or not."""
    try:
        subprocess.check_output(f"ray get_head_ip {cluster_yaml}", shell=True, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def start_cluster_if_needed(cluster_yaml: str) -> None:
    """Starts the cluster if it isn't already running."""
    if not is_cluster_running(cluster_yaml):
        subprocess.run(f"ray up {cluster_yaml} -y", shell=True)


def generate_cluster_config_path(config_filename: str) -> str:
    """Generate the path to the rsynced config on the cluster"""
    return f"{CLUSTER_CONFIG_DIR}/{current_timestamp()}{os.path.basename(config_filename)}"
