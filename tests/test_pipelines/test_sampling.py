import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.order(after="test_rasterize.py::test_rasterize_feature_with_resolution")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "sampling_fraction",
        "sampling_fraction_erosion",
        "sampling_block_number",
        "sampling_block_fraction",
        "sampling_grid",
        pytest.param("sampling_chain", marks=pytest.mark.chain),
    ],
)
def test_sampling_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("sampling", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path)
