"""
Testing training pipelines
"""
import os

import joblib
import pytest

from eogrow.core.config import interpret_config_from_path
from eogrow.pipelines.training import ClassificationTrainingPipeline
from eogrow.utils.testing import create_folder_dict


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "training")


@pytest.mark.chain
@pytest.mark.order(after="test_merge_samples.py::test_merge_samples_pipeline")
@pytest.mark.parametrize(
    "experiment_name, num_classes",
    [("lgbm_training_no_filter", 6), ("lgbm_training_label_filter", 3)],
)
def test_training_pipeline_random_split(folders, experiment_name, num_classes):
    config_path = os.path.join(folders["config_folder"], experiment_name + ".json")
    pipeline = ClassificationTrainingPipeline.from_raw_config(interpret_config_from_path(config_path))
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
