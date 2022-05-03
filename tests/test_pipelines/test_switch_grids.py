"""
Tests for grid switching
"""
import os
import shutil

import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="batch_folders")
def batch_folders_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, os.path.join("switch_grids", "batch"))


@pytest.fixture(scope="session", name="utm_folders")
def utm_folders_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, os.path.join("switch_grids", "utm"))


@pytest.fixture(scope="session", name="batch_grid")
def batch_grid_fixture(project_folder):
    """Cached grid file can't be created on the fly therefore it is committed in the input-data folder. This fixture
    just copies it into the cache folder."""
    grid_path = os.path.join(project_folder, "input-data", "batch_grid.gpkg")

    cache_folder = os.path.join(project_folder, "cache")
    os.makedirs(cache_folder, exist_ok=True)
    cache_grid_path = os.path.join(cache_folder, "grid_batch_area_BatchAreaManager___1_10.0_3_7.gpkg")

    shutil.copyfile(grid_path, cache_grid_path)
    return cache_grid_path


@pytest.mark.parametrize(
    "experiment_name",
    [
        "switch1",
        "switch2",
    ],
)
def test_batch_grid_switching(experiment_name, batch_folders, batch_grid):
    run_and_test_pipeline(experiment_name, **batch_folders)


@pytest.mark.parametrize(
    "experiment_name",
    [
        "switch1",
        "switch2",
    ],
)
def test_utm_grid_switching(experiment_name, utm_folders):
    run_and_test_pipeline(experiment_name, **utm_folders)
