"""
Tests for prediction pipeline
"""
import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "mapping")


@pytest.mark.order(after=["test_prediction.py::test_prediction_pipeline"])
@pytest.mark.parametrize("experiment_name", ["mapping_ref", pytest.param("mapping_pred", marks=pytest.mark.chain)])
def test_mapping_pipeline_on_reference_data(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
