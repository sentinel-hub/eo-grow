import os
import shutil

import pytest

from eogrow.core.area import CustomGridAreaManager
from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="folders")
def folders_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "split_grid")


@pytest.fixture(scope="session", name="batch_grid")
def batch_grid_fixture(project_folder):
    """Cached grid file can't be created on the fly therefore it is committed in the input-data folder. This fixture
    just copies it into the cache folder."""
    grid_path = os.path.join(project_folder, "input-data", "batch_grid.gpkg")

    cache_folder = os.path.join(project_folder, "cache")
    os.makedirs(cache_folder, exist_ok=True)
    cache_grid_path = os.path.join(cache_folder, "BatchAreaManager_batch_area_1_10.0_3_7.gpkg")

    shutil.copyfile(grid_path, cache_grid_path)
    return cache_grid_path


@pytest.mark.parametrize("experiment_name", ["split_batch", "split_utm"])
def test_grid_splitting(experiment_name, folders, batch_grid, test_storage_manager):
    run_and_test_pipeline(experiment_name, **folders, folder_key="output_folder")

    # check the output file is compatible with custom grid area managers
    new_area_manager = CustomGridAreaManager.from_raw_config(
        {"grid_folder_key": "temp", "grid_filename": "new_grid.gpkg", "name_column": "eopatch_name"},
        storage=test_storage_manager,
    )
    new_area_manager.get_grid()
