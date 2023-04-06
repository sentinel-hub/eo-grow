import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain
@pytest.mark.order(after="test_features.py::test_features_pipeline")
@pytest.mark.parametrize(
    "experiment_name, reset_folder",
    [("merge_features_samples", True), ("merge_reference_samples", False)],
)
def test_merge_samples_pipeline(config_and_stats_paths, experiment_name, reset_folder):
    config_path, stats_path = config_and_stats_paths("merge_samples", experiment_name)
    run_config(config_path, reset_output_folder=reset_folder)
    compare_content(config_path, stats_path)
