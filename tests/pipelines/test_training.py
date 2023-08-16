import os

import joblib
import pytest

from eogrow.pipelines.training import ClassificationTrainingPipeline

pytestmark = pytest.mark.integration


@pytest.mark.order(after="test_merge_samples.py::test_merge_samples_pipeline")
@pytest.mark.parametrize(
    ("experiment_name", "num_classes"),
    [pytest.param("lgbm_training_no_filter", 6, marks=pytest.mark.chain), ("lgbm_training_label_filter", 3)],
)
def test_training_pipeline_random_split(config_and_stats_paths, experiment_name, num_classes):
    config_path, _ = config_and_stats_paths("training", experiment_name)

    pipeline = ClassificationTrainingPipeline.from_path(config_path)
    pipeline.run()  # not run with `run_config` because it checks EOExecutor logs, but this one has none
    config = pipeline.config

    folder = pipeline.storage.get_folder("models", full_path=True)
    model_path = os.path.join(folder, config.model_filename)

    assert os.path.isfile(model_path)
    model = joblib.load(model_path)
    assert model.n_features_ == 192
    assert model.n_classes_ == num_classes

    if config.preprocessing and config.preprocessing.label_encoder_filename:
        encoder_path = os.path.join(folder, config.preprocessing.label_encoder_filename)
        assert os.path.isfile(encoder_path)
