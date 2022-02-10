"""
Testing training pipelines
"""
import os

import pytest
import joblib

from eogrow.core.config import Config
from eogrow.pipelines.training import ClassificationTrainingPipeline


@pytest.mark.chain
@pytest.mark.order(after="test_merge_samples.py::test_merge_samples_pipeline")
@pytest.mark.parametrize(
    "config_name, num_classes",
    [
        ("lgbm_training_config_no_filter.json", 6),
        ("lgbm_training_config_label_filter.json", 3),
    ],
)
def test_training_pipeline_random_split(config_folder, config_name, num_classes):
    config_path = os.path.join(config_folder, config_name)
    config = Config.from_path(config_path)

    pipeline = ClassificationTrainingPipeline(config)
    config = pipeline.config
    pipeline.run()

    folder = pipeline.storage.get_folder("models", full_path=True)
    model_path = os.path.join(folder, config.model_filename)
    assert os.path.isfile(model_path)
    if config.preprocessing and config.preprocessing.label_encoder_filename:
        encoder_path = os.path.join(folder, config.preprocessing.label_encoder_filename)
        assert os.path.isfile(encoder_path)

    # TODO: better checks could be implemented in ContentTester
    model = joblib.load(model_path)
    assert model.n_features_ == 192
    assert model.n_classes_ == num_classes
