"""Implements a base training pipeline and a LGBM specialized classification and regression model training pipelines."""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Literal, Optional

import fs
import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from pydantic import Field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..utils.validators import ensure_storage_key_presence

LOGGER = logging.getLogger(__name__)


class RandomTrainTestSplitSchema(BaseSchema):
    random_state: int = Field(
        42,
        description="Seed used in data splitter (either for scikit.learn.train_test_split or for scikit.utils.shuffle.",
    )
    # random training/testing split parameters
    train_size: float = Field(ge=0, le=1, description="Training size value (0.8 = 80/20 split for training/testing).")


class BaseTrainingPipeline(Pipeline, metaclass=abc.ABCMeta):
    """A base pipeline for training an ML model

    This class has a few abstract methods which have to be implemented. But in general all public methods are designed
    in a way that you can override them in a child class
    """

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(description="The storage manager key pointing to the model training data.")
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        model_folder_key: str = Field(
            description="The storage manager key pointing to the folder where the model will be saved."
        )
        _ensure_model_folder_key = ensure_storage_key_presence("model_folder_key")

        train_features: List[str] = Field(
            description="A list of feature filenames to join into training features in the given order."
        )

        train_reference: str = Field(description="Name of file where the reference data is stored.")

        train_test_split: RandomTrainTestSplitSchema

        model_parameters: Dict[str, Any] = Field(
            default_factory=dict, description="Parameters to be provided to the model"
        )
        model_filename: str
        patch_list: None = None
        input_patch_file: None = None
        skip_existing: Literal[False] = False

    config: Schema

    def run_procedure(self) -> tuple[list[str], list[str]]:
        """The main pipeline procedure

        1. Prepares data. Output serves as input to both the training method and scoring method, so separation of
           training and testing data should be done within the object.
        2. Train model
        3. Save model
        4. Evaluate model
        """
        LOGGER.info("Preparing data")
        prepared_data = self.prepare_data()

        LOGGER.info("Training. This could take a while.")
        model = self.train_model(prepared_data)

        LOGGER.info("Saving.")
        self.save_model(model)

        LOGGER.info("Scoring results.")
        self.score_results(prepared_data, model)

        return [], []

    def prepare_data(self) -> dict:
        """Loads and preprocesses data."""
        features = self._collect_features()
        reference = self._collect_reference()

        features, reference = self.preprocess_data(features, reference)

        data_split = self.train_test_split(features, reference)
        return dict(zip(["features_train", "features_test", "reference_train", "reference_test"], data_split))

    def _collect_features(self) -> np.ndarray:
        """Prepares features"""
        features = []
        LOGGER.info("Reading input features")

        for feature_name in self.config.train_features:
            array = self._read_array(feature_name)
            array = self._reshape_array(array)
            features.append(array)

        return np.concatenate(features, axis=1)

    def _collect_reference(self) -> np.ndarray:
        """Prepares reference"""
        LOGGER.info("Reading input reference")
        return self._read_array(self.config.train_reference)

    def preprocess_data(self, features: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preforms filtering and other preprocessing before splitting data."""
        return features, reference.ravel()

    def train_test_split(
        self, features: np.ndarray, reference: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes a random train-test split

        Order is train-features test-features train-reference test-reference.
        """
        config = self.config.train_test_split
        LOGGER.info("Making a random train-test split, using %s of data for training.", config.train_size)
        return train_test_split(features, reference, train_size=config.train_size, random_state=config.random_state)

    def _read_array(self, filename: str) -> np.ndarray:
        """Read numpy array from the training data folder given a filename."""
        folder = self.storage.get_folder(self.config.input_folder_key)
        if not filename.endswith(".npy"):
            filename = f"{filename}.npy"
        path = fs.path.combine(folder, filename)

        with self.storage.filesystem.openbin(path, "r") as file_handle:
            return np.load(file_handle)

    @staticmethod
    def _reshape_array(array: np.ndarray) -> np.ndarray:
        """Reshape numpy array into 2D representation suitable for fitting a model."""
        shape = array.shape
        return array.reshape((shape[0], np.prod(shape[1:])))

    @abc.abstractmethod
    def train_model(self, prepared_data: dict) -> object:
        """Trains the model on the data."""

    def save_model(self, model: object) -> None:
        """Saves the resulting model."""
        self._dump_object(self.config.model_filename, model)

    def _dump_object(self, filename: str, object_instance: object) -> None:
        """Dumps an object instance into models folder"""
        LOGGER.info("Saving to %s", filename)
        folder = self.storage.get_folder(self.config.model_folder_key)
        path = fs.path.combine(folder, filename)

        with self.storage.filesystem.openbin(path, "w") as file_handle:
            joblib.dump(object_instance, file_handle)

    @abc.abstractmethod
    def score_results(self, prepared_data: dict, model: Any) -> None:
        """Scores the resulting model and reports the metrics into the log files."""

    def predict(self, model: Any, features: np.ndarray) -> np.ndarray:  # pylint: disable=no-self-use
        """Evaluates model on features. Should be overridden for models with a different interface."""
        return model.predict(features)


class ClassificationPreprocessSchema(BaseSchema):
    label_encoder_filename: Optional[str] = Field(
        description="If specified uses a label encoder and saves it under specified name."
    )

    filter_classes: List[int] = Field(
        default_factory=list,
        description=(
            "Specify IDs of classes that are going to be used for training. If empty, all the classes will be used."
        ),
    )


class ClassificationTrainingPipeline(BaseTrainingPipeline):
    """A base pipeline for training an ML classifier. Uses LGBMClassifier by default."""

    class Schema(BaseTrainingPipeline.Schema):
        preprocessing: Optional[ClassificationPreprocessSchema]

    config: Schema

    def preprocess_data(self, features: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preforms filtering and other preprocessing before splitting data."""
        config = self.config.preprocessing
        reference = reference.ravel()
        if config is None:
            return features, reference

        if config.filter_classes:
            filtration = np.isin(reference, config.filter_classes).squeeze()
            features, reference = features[filtration, ...], reference[filtration]

        if config.label_encoder_filename:
            LOGGER.info("Applying label encoder to labels")
            label_encoder = LabelEncoder()
            label_encoder.fit(np.unique(reference))
            reference = label_encoder.transform(reference)

            self._dump_object(config.label_encoder_filename, label_encoder)

        return features, reference

    def train_model(self, prepared_data: dict) -> object:
        train_features = prepared_data["features_train"]
        train_reference = prepared_data["reference_train"]
        lgbm_classifier = LGBMClassifier(**self.config.model_parameters)
        lgbm_classifier.fit(train_features, train_reference)

        return lgbm_classifier

    def score_results(self, prepared_data: dict, model: Any) -> None:
        test_features = prepared_data["features_test"]
        test_reference = prepared_data["reference_test"]
        predictions = self.predict(model, test_features)

        accuracy = accuracy_score(test_reference, predictions)
        f1_value = f1_score(test_reference, predictions, average="weighted")
        recall = recall_score(test_reference, predictions, average="weighted")
        precision = precision_score(test_reference, predictions, average="weighted")

        LOGGER.info("Accuracy:\t %.4f", accuracy)
        LOGGER.info("F1 score:\t %.4f", f1_value)
        LOGGER.info("Recall:\t %.4f", recall)
        LOGGER.info("Precision:\t %.4f", precision)


class RegressionTrainingPipeline(BaseTrainingPipeline):
    """A base pipeline for training an ML regressor. Uses LGBMRegressor by default."""

    def train_model(self, prepared_data: dict) -> object:
        train_features = prepared_data["features_train"]
        train_reference = prepared_data["reference_train"]
        lgbm_classifier = LGBMRegressor(**self.config.model_parameters)
        lgbm_classifier.fit(train_features, train_reference)

        return lgbm_classifier

    def score_results(self, prepared_data: dict, model: Any) -> None:
        test_features = prepared_data["features_test"]
        test_reference = prepared_data["reference_test"]
        predictions = self.predict(model, test_features)

        r2_value = r2_score(test_reference, predictions)
        mse = mean_squared_error(test_reference, predictions)
        rmse = mean_squared_error(test_reference, predictions, squared=False)
        mae = mean_absolute_error(test_reference, predictions)

        LOGGER.info("Mean Square Error:\t %.4f", mse)
        LOGGER.info("Root Mean Square Error:\t %.4f", rmse)
        LOGGER.info("Mean Average Error:\t %.4f", mae)
        LOGGER.info("R2 Score:\t %.4f", r2_value)
