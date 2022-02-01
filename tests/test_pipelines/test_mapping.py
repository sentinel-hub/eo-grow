"""
Tests for prediction pipeline
"""
import os

import pytest

from eogrow.core.config import Config
from eogrow.utils.testing import run_and_test_pipeline


@pytest.mark.chain
@pytest.mark.order(after=["test_rasterize.py::test_rasterize_pipeline_features"])
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("mapping_config_ref.json", "mapping_stats_ref.json"),
    ],
)
def test_mapping_pipeline_on_reference_data(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.order(after=["test_prediction.py::test_prediction_pipeline"])
@pytest.mark.parametrize("config_name, stats_name", [("mapping_config_pred.json", "mapping_stats_pred.json")])
def test_mapping_pipeline_on_predictions(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)
