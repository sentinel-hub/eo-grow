import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain()
@pytest.mark.parametrize("experiment_name", ["testing", "testing_new", "timestamps_only", "timestamps_only_new"])
def test_features_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("testing", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path)
