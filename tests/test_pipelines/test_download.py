"""
Unit tests for DownloadPipeline
"""
import os

from pydantic import ValidationError
import pytest

from eogrow.core.config import Config
from eogrow.utils.testing import run_and_test_pipeline


@pytest.mark.chain
@pytest.mark.order(before=["test_download_pipeline", "test_download_step_of_chain"])
def test_preparation(storage):
    """Cleans the test project folder"""


@pytest.mark.order(before="test_download_step_of_chain")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("download_l1c_q1.json", "download_l1c_q1_stats.json"),
        ("download_l1c_q1_dn.json", "download_l1c_q1_dn_stats.json"),
        ("download_l1c_q1_dn_rescaled.json", "download_l1c_q1_dn_rescaled_stats.json"),
        ("download_l2a.json", "download_stats_l2a.json"),
        ("download_season.json", "download_season_stats.json"),
        ("download_custom.json", "download_custom_stats.json"),
        ("download_q3.json", "download_q3_stats.json"),
        ("download_dem.json", "download_dem.json"),
    ],
)
def test_download_pipeline(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.chain
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("download_l1c_yearly.json", "download_l1c_yearly_stats.json"),
    ],
)
def test_download_step_of_chain(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.parametrize(
    "config_name",
    [
        "download_custom_raise.json",  # end_date < start_date
    ],
)
def test_validation_error(config_folder, config_name):
    with pytest.raises(ValidationError):
        run_and_test_pipeline(config_folder, config_name, "", "", "data")
