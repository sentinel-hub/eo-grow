"""
Testing pipeline for exporting maps
"""
import os

import pytest

from eogrow.core.config import Config
from eogrow.utils.testing import run_and_test_pipeline


@pytest.mark.order(after="test_download.py::test_download_step_of_chain")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("export_maps_mask_config.json", "export_maps_stats_mask.json"),
    ],
)
def test_export_maps_pipeline(config_folder, stats_folder, config_name, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.chain
@pytest.mark.order(after="test_download.py::test_download_step_of_chain")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("export_maps_data_config.json", "export_maps_stats_data.json"),
    ],
)
def test_export_maps_pipeline_in_chain(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)
