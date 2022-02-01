"""
Unit tests for sampling pipeline
"""
import os
import pytest

from eogrow.core.config import Config
from eogrow.utils.testing import run_and_test_pipeline


@pytest.mark.order(after="test_rasterize.py::test_rasterize_pipeline")
@pytest.mark.parametrize(
    "config_name,stats_name",
    [
        ("sampling_fraction_config.json", "sampling_fraction_stats.json"),
        ("sampling_block_number_config.json", "sampling_block_number_stats.json"),
        ("sampling_block_fraction_config.json", "sampling_block_fraction_stats.json"),
        ("sampling_grid_config.json", "sampling_blocks_number_stats.json"),
    ],
)
def test_sampling_pipeline(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.chain
@pytest.mark.order(after="test_rasterize.py::test_rasterize_pipeline")
@pytest.mark.parametrize(
    "config_name,stats_name",
    [
        ("sampling_config.json", "sampling_stats.json"),
    ],
)
def test_sampling_pipeline(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)
