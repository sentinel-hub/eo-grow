"""Defines task needed in prediction pipelines."""

from __future__ import annotations

import abc
from typing import Any, Callable

import fs
import joblib
import numpy as np
from fs.base import FS

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import Feature
from eolearn.core.utils.fs import pickle_fs, unpickle_fs


class BasePredictionTask(EOTask, metaclass=abc.ABCMeta):
    """Base predictions task streamlining data preprocessing before prediction"""

    def __init__(
        self,
        *,
        model_path: str,
        filesystem: FS,
        input_features: list[Feature],
        mask_feature: Feature,
        output_feature: Feature,
        output_dtype: np.dtype | None,
    ):
        """
        :param model_path: A file path to the model. The path is relative to the filesystem object.
        :param filesystem: A filesystem object.
        :param input_features: List of features containing input for the model, which are concatenated in given order
        :param mask_feature: Mask specifying which points are to be predicted
        :param output_feature: Feature into which predictions are saved
        :param mp_lock: If predictions should be executed with a multiprocessing lock
        """
        self.model_path = model_path
        self._model = None
        self.pickled_filesystem = pickle_fs(filesystem)

        self.input_features = input_features
        self.mask_feature = mask_feature
        self.output_feature = output_feature
        self.output_dtype = output_dtype

    def process_data(self, eopatch: EOPatch, mask: np.ndarray) -> np.ndarray:
        """Masks and reshapes data into a form suitable for the model"""
        all_features = []
        for ftype, fname in self.input_features:
            array = eopatch[ftype, fname]

            if ftype.is_timeless():
                all_features.append(array[mask, :])
            else:
                array = array[:, mask, :]
                time, pixels, depth = array.shape
                array = np.moveaxis(array, 0, 1)
                all_features.append(array.reshape(pixels, time * depth))

        return np.concatenate(all_features, axis=-1)

    @property
    def model(self) -> Any:
        """Implements lazy loading that gets around filesystem issues"""
        if self._model is None:
            filesystem = unpickle_fs(self.pickled_filesystem)

            with filesystem.openbin(self.model_path, "r") as file_handle:
                self._model = joblib.load(file_handle)

        return self._model

    def apply_predictor(
        self, predictor: Callable, processed_features: np.ndarray, return_on_empty: np.ndarray | None = None
    ) -> np.ndarray:
        """Helper function that applies the predictor according to the mp_lock settings"""
        if processed_features.shape[0] == 0 and return_on_empty is not None:
            return return_on_empty

        predictions: np.ndarray = predictor(processed_features)
        return predictions.astype(self.output_dtype) if self.output_dtype else predictions

    @abc.abstractmethod
    def add_predictions(self, eopatch: EOPatch, processed_features: np.ndarray, mask: np.ndarray) -> EOPatch:
        """Runs the model prediction on given features and adds them to the eopatch. Must reverse mask beforehand."""

    @staticmethod
    def transform_to_feature_form(predictions: np.ndarray, mask: np.ndarray, no_value: float | int = 0) -> np.ndarray:
        """Transforms an array of predictions into an EOPatch suitable array, making sure to reverse the masking"""
        full_predictions = np.full((*mask.shape, predictions.shape[-1]), dtype=predictions.dtype, fill_value=no_value)
        full_predictions[mask, :] = predictions
        return full_predictions

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Run model on input features and save predictions to eopatch"""

        some_feature = self.input_features[0]
        mask_size = eopatch.get_spatial_dimension(*some_feature)
        mask = np.squeeze(eopatch[self.mask_feature], axis=-1) if self.mask_feature else np.ones(mask_size)
        mask = mask.astype(bool)

        preprocessed_features = self.process_data(eopatch, mask)
        return self.add_predictions(eopatch, preprocessed_features, mask)


class ClassificationPredictionTask(BasePredictionTask):
    """Uses a classification model to produce predictions for given input features"""

    def __init__(
        self,
        *,
        label_encoder_filename: str | None,
        output_probability_feature: Feature | None = None,
        **kwargs: Any,
    ):
        """
        :param label_encoder_filename: Name of file containing the label encoder with which to decode predictions. The
            file should be in the same folder as the model.
        :param output_probability_feature: If specified saves pseudo-probabilities into given feature.
        :param kwargs: Parameters of `BasePredictionTask`
        """
        self.label_encoder_filename = label_encoder_filename
        self._label_encoder = None
        self.output_probability_feature = output_probability_feature
        super().__init__(**kwargs)

    @property
    def label_encoder(self) -> Any:
        """Implements lazy loading that gets around filesystem issues"""
        if self._label_encoder is None and self.label_encoder_filename is not None:
            filesystem = unpickle_fs(self.pickled_filesystem)

            model_folder = fs.path.dirname(self.model_path)
            label_encoder_path = fs.path.join(model_folder, self.label_encoder_filename)

            with filesystem.openbin(label_encoder_path, "r") as file_handle:
                self._label_encoder = joblib.load(file_handle)

        return self._label_encoder

    def add_predictions(self, eopatch: EOPatch, processed_features: np.ndarray, mask: np.ndarray) -> EOPatch:
        """Runs the model prediction on given features and adds them to the eopatch

        If specified also adds probability scores and uses a label encoder.
        """
        predictions = self.apply_predictor(self.model.predict, processed_features, np.zeros((0,), dtype=np.uint8))

        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        predictions = predictions[..., np.newaxis]
        eopatch[self.output_feature] = self.transform_to_feature_form(predictions, mask)

        if self.output_probability_feature is not None:
            probabilities = self.apply_predictor(self.model.predict_proba, processed_features)
            eopatch[self.output_probability_feature] = self.transform_to_feature_form(probabilities, mask)

        return eopatch


class RegressionPredictionTask(BasePredictionTask):
    """Computes values and scores given an input model and eopatch feature name"""

    def __init__(
        self,
        *,
        clip_predictions: tuple[float, float] | None,
        **kwargs: Any,
    ):
        """
        :param clip_predictions: If given the task also clips predictions to the specified interval.
        :param kwargs: Parameters of `BasePredictionTask`
        """
        self.clip_predictions = clip_predictions
        super().__init__(**kwargs)

    def add_predictions(self, eopatch: EOPatch, processed_features: np.ndarray, mask: np.ndarray) -> EOPatch:
        """Runs the model prediction on given features and adds them to the eopatch. Must reverse mask beforehand."""

        predictions = self.apply_predictor(self.model.predict, processed_features, np.zeros((0,), dtype=np.float32))
        predictions = predictions[..., np.newaxis]
        if self.clip_predictions is not None:
            predictions = predictions.clip(*self.clip_predictions)
        eopatch[self.output_feature] = self.transform_to_feature_form(predictions, mask)

        return eopatch
