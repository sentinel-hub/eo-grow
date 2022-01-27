"""
Testing merging pipeline
"""
import pytest

from eogrow.utils.testing import run_and_test_pipeline


@pytest.mark.chain
@pytest.mark.order(after="test_features.py::test_features_pipeline_in_chain")
@pytest.mark.parametrize(
    "config_name, stats_name, reset_folder",
    [
        ("merge_features_samples_config.json", "merge_features_samples_stats.json", True),
        ("merge_reference_samples_config.json", "merge_reference_samples_stats.json", False),
    ],
)
def test_merge_samples_pipeline(config_folder, config_name, stats_folder, stats_name, reset_folder):
    run_and_test_pipeline(
        config_folder, config_name, stats_folder, stats_name, "training_data", reset_folder=reset_folder
    )
