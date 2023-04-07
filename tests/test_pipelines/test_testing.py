import pytest

from eogrow.utils.testing import extract_output_folder, new_compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain
@pytest.mark.parametrize("experiment_name", ["testing", "timestamps_only"])
def test_features_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("testing", experiment_name)
    run_config(config_path)
    new_compare_content(extract_output_folder(config_path), stats_path)
