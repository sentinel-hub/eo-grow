import pytest

from eogrow.utils.testing import extract_output_folder, new_compare_content, run_config

pytestmark = pytest.mark.integration


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
def test_prediction_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("prediction", experiment_name)
    run_config(config_path)
    new_compare_content(extract_output_folder(config_path), stats_path)
