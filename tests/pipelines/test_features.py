import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.order(after="test_sampling.py::test_sampling_pipeline")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "features_mosaicking",
        "features_dtype",
        pytest.param("features_on_sampled_data", marks=pytest.mark.chain),
    ],
)
def test_features_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("features", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path)
