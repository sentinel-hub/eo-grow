"""
Tests for prediction pipeline
"""
import os

import pytest

from eogrow.core.config import Config
from eogrow.utils.testing import run_and_test_pipeline


@pytest.mark.order(after="test_training.py::test_training_pipeline_random_split")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("prediction_config.json", "prediction_stats.json"),
        ("prediction_with_encoder_config.json", "prediction_with_encoder_stats.json"),
    ],
)
def test_prediction_pipeline(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.chain
@pytest.mark.order(after="test_training.py::test_training_pipeline_random_split")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("prediction_chain_config.json", "prediction_chain_stats.json"),
    ],
)
def test_prediction_chain(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)
