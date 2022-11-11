"""
Unit tests for FeaturesPipeline
"""
import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "features")


@pytest.mark.order(after="test_sampling.py::test_sampling_pipeline")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "features_interpolation",
        "features_mosaicking_custom",
        "features_on_rescaled_dn",
        "features_mosaicking",
        "features_dtype",
        pytest.param("features_on_sampled_data", marks=pytest.mark.chain),
    ],
)
def test_features_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
