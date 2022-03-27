"""
Testing merging pipeline
"""
import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "merge_samples")


@pytest.mark.chain
@pytest.mark.order(after="test_features.py::test_features_pipeline_in_chain")
@pytest.mark.parametrize(
    "experiment_name, reset_folder",
    [("merge_features_samples", True), ("merge_reference_samples", False)],
)
def test_merge_samples_pipeline(experiment_name, reset_folder, folders):
    run_and_test_pipeline(experiment_name, **folders, reset_folder=reset_folder)
