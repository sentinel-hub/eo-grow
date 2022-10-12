"""
Testing pipeline for exporting maps
"""
import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "export_maps")


@pytest.mark.order(after="test_download.py::test_download_step_of_chain")
@pytest.mark.parametrize("experiment_name", ["export_maps_mask", "export_maps_mask_local_copy"])
def test_export_maps_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)


@pytest.mark.chain
@pytest.mark.order(after="test_download.py::test_download_step_of_chain")
@pytest.mark.parametrize("experiment_name", ["export_maps_data"])
def test_export_maps_pipeline_in_chain(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
