"""
Unit tests for sampling pipeline
"""
import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "sampling")


@pytest.mark.order(after="test_rasterize.py::test_rasterize_pipeline")
@pytest.mark.parametrize(
    "experiment_name",
    ["sampling_fraction", "sampling_block_number", "sampling_block_fraction", "sampling_grid"],
)
def test_sampling_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)


@pytest.mark.chain
@pytest.mark.order(after="test_rasterize.py::test_rasterize_pipeline")
@pytest.mark.parametrize("experiment_name", ["sampling_chain"])
def test_sampling_chain(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
