import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain
@pytest.mark.order(after="test_features.py::test_features_pipeline")
def test_merge_samples_pipeline(config_and_stats_paths):
    config_path, stats_path = config_and_stats_paths("merge_samples", "merge_features_samples")
    output_path = run_config(config_path, reset_output_folder=True)
    compare_content(output_path, stats_path)

    config_path, stats_path = config_and_stats_paths("merge_samples", "merge_reference_samples")
    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)
