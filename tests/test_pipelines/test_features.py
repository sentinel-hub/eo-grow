"""
Unit tests for FeaturesPipeline
"""
import os
import pytest

from eogrow.utils.testing import run_and_test_pipeline
from eogrow.core.config import Config


@pytest.mark.order(after="test_download.py::test_download_pipeline")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("features_config_interpolation.json", "features_stats_interpolation.json"),
        ("features_config_mosaicking_custom.json", "features_stats_mosaicking_custom.json"),
        ("features_on_rescaled_dn.json", "features_on_rescaled_dn.json"),
        ("features_config_mosaicking.json", "features_stats_mosaicking.json"),
    ],
)
def test_features_pipeline(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.chain
@pytest.mark.order(after="test_sampling.py::test_sampling_pipeline")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("features_on_sampled_data.json", "features_on_sampled_data.json"),
    ],
)
def test_features_pipeline_in_chain(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)
