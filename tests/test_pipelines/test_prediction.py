import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "prediction")


@pytest.mark.order(after="test_training.py::test_training_pipeline_random_split")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "prediction",
        "prediction_dtype",
        "prediction_with_encoder",
        pytest.param("prediction_chain", marks=pytest.mark.chain),
    ],
)
def test_prediction_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
