"""
Unit tests for DownloadPipeline
"""
import pytest
from pydantic import ValidationError

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "download_and_batch")


@pytest.mark.chain
@pytest.mark.order(before=["test_download_pipeline"])
def test_preparation(storage):
    """Cleans the test project folder"""


@pytest.mark.parametrize(
    "experiment_name",
    [
        "download_l1c_q1",
        "download_l1c_q1_dn",
        "download_l1c_q1_dn_rescaled",
        "download_l2a",
        "download_season",
        "download_custom",
        "download_q3",
        "download_dem",
        pytest.param("download_l1c_yearly", marks=pytest.mark.chain),
    ],
)
def test_download_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)


@pytest.mark.parametrize("experiment_name", ["download_custom_raise"])
def test_validation_error(experiment_name, folders):
    with pytest.raises(ValidationError):
        run_and_test_pipeline(experiment_name, **folders)
