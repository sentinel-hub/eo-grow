import pytest

from eogrow.core.area import CustomGridAreaManager
from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("preparation_config", "config"),
    [pytest.param("dummy_data_utm", "split_utm", marks=pytest.mark.chain)],
)
def test_grid_splitting(config_and_stats_paths, preparation_config, config, test_storage_manager):
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
