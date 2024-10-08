import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain
@pytest.mark.order(before=["test_download_pipeline"])
@pytest.mark.usefixtures("storage")
def test_preparation():
    """Cleans the test project folder"""


@pytest.mark.parametrize(
    "experiment_name",
    [
        "download_l1c_q1_dn_rescaled",
        "download_custom_collection",
        "download_dem",
        pytest.param("download_l1c_yearly", marks=pytest.mark.chain),
    ],
)
def test_download_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("download_and_batch", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path)
