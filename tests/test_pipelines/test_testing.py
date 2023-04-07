import pytest

from eogrow.utils.testing import compare_content, extract_output_folder, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain
@pytest.mark.parametrize("experiment_name", ["testing", "timestamps_only"])
def test_features_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("testing", experiment_name)
    run_config(config_path)
    compare_content(extract_output_folder(config_path), stats_path)
