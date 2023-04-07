import os
import shutil

import pytest

from eogrow.core.area import CustomGridAreaManager
from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


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


@pytest.mark.parametrize(
    "preparation_config, config",
    [("dummy_data_batch", "split_batch"), pytest.param("dummy_data_utm", "split_utm", marks=pytest.mark.chain)],
)
def test_grid_splitting(config_and_stats_paths, preparation_config, config, batch_grid, test_storage_manager):
    preparation_config_path, _ = config_and_stats_paths("split_grid", preparation_config)
    config_path, stats_path = config_and_stats_paths("split_grid", config)

    run_config(preparation_config_path)
    output_path = run_config(config_path, output_folder_key="output_folder")
    compare_content(output_path, stats_path)

    # check the output file is compatible with custom grid area managers
    new_area_manager = CustomGridAreaManager.from_raw_config(
        {"grid_folder_key": "temp", "grid_filename": "new_grid.gpkg", "name_column": "eopatch_name"},
        storage=test_storage_manager,
    )
    new_area_manager.get_grid()
