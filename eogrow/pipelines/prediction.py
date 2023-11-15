"""Implements a base prediction pipeline and a LGBM specialized classification and regression pipelines."""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import fs
import numpy as np
from pydantic import Field

from eolearn.core import EONode, EOWorkflow, FeatureType, LoadTask, MergeEOPatchesTask, OverwritePermission, SaveTask
from eolearn.core.types import Feature

from ..core.pipeline import Pipeline
from ..tasks.prediction import ClassificationPredictionTask, RegressionPredictionTask
from ..types import PatchList
from ..utils.filter import get_patches_with_missing_features
from ..utils.validators import (
    ensure_defined_together,
    ensure_storage_key_presence,
    optional_field_validator,
    parse_dtype,
)


class BasePredictionPipeline(Pipeline, metaclass=abc.ABCMeta):
    """Pipeline to load a model and run prediction on EOPatches"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder of the model input data."
        )
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        input_features: List[Feature] = Field(
            description=(
                "List of features of form `[(feature_type, feature_name)]` specifying which features are model input in"
                " the correct order"
            )
        )

        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the prediction pipeline."
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        dtype: Optional[np.dtype] = Field(
            description="Casts the result to desired type. Uses predictor output type by default."
        )
        _parse_dtype = optional_field_validator("dtype", parse_dtype, pre=True)

        prediction_mask_feature_name: Optional[str] = Field(
            description="Name of `MASK_TIMELESS` feature which defines which areas will be predicted"
        )
        prediction_mask_folder_key: Optional[str]
        _ensure_mask_feature_key = ensure_defined_together("prediction_mask_feature_name", "prediction_mask_folder_key")

        model_folder_key: str = Field(
            description="The storage manager key pointing to the folder of the model used in the prediction pipeline."
        )
        _ensure_model_folder_key = ensure_storage_key_presence("model_folder_key")

    config: Schema

    @abc.abstractmethod
    def _get_output_features(self) -> list[Feature]:
        """Lists all features that are to be saved upon the pipeline completion"""

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of specified output features"""
        output_features = self._get_output_features()
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            output_features,
            check_timestamps=False,
        )

    def build_workflow(self) -> EOWorkflow:
        """Workflow handling the prediction for eopatches.
        The workflow allows to add smoothing and custom thresholding to the predicted scores
        """
        preparation_node = self._get_data_preparation_node()
        predictions_node = self._get_prediction_node(preparation_node)
        saving_node = self._get_saving_node(predictions_node)

        return EOWorkflow.from_endnodes(saving_node)

    def _get_data_preparation_node(self) -> EONode:
        """Returns nodes containing for loading and preparing the data as well as the endpoint tasks"""
        features_load_node = EONode(
            LoadTask(
                self.storage.get_folder(self.config.input_folder_key),
                filesystem=self.storage.filesystem,
                features=self.config.input_features,
            )
        )

        if not self.config.prediction_mask_folder_key or not self.config.prediction_mask_feature_name:
            return features_load_node

        mask_load_node = EONode(
            LoadTask(
                self.storage.get_folder(self.config.prediction_mask_folder_key),
                filesystem=self.storage.filesystem,
                features=[(FeatureType.MASK_TIMELESS, self.config.prediction_mask_feature_name)],
            )
        )

        return EONode(MergeEOPatchesTask(), inputs=[features_load_node, mask_load_node])

    @abc.abstractmethod
    def _get_prediction_node(self, previous_node: EONode) -> EONode:
        """Returns nodes for using the model on the prepared data and the endpoints of the dependencies"""

    def _get_saving_node(self, previous_node: EONode) -> EONode:
        """Returns nodes for finalizing and saving features"""
        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            features=self._get_output_features(),
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            use_zarr=self.storage.config.use_zarr,
        )

        return EONode(save_task, inputs=[previous_node])


class RegressionPredictionPipeline(BasePredictionPipeline):
    class Schema(BasePredictionPipeline.Schema):
        output_feature_name: str
        model_filename: str = Field(description="A filename of a regression model to be used for prediction.")
        clip_predictions: Optional[Tuple[float, float]] = Field(
            description="Clip values of predictions to specified interval"
        )

    config: Schema

    def _get_output_features(self) -> list[Feature]:
        return [(FeatureType.DATA_TIMELESS, self.config.output_feature_name)]

    def _get_prediction_node(self, previous_node: EONode) -> EONode:
        model_path = fs.path.join(self.storage.get_folder(self.config.model_folder_key), self.config.model_filename)
        prediction_task = RegressionPredictionTask(
            model_path=model_path,
            filesystem=self.storage.filesystem,
            input_features=self.config.input_features,
            mask_feature=_optional_typed_feature(FeatureType.MASK_TIMELESS, self.config.prediction_mask_feature_name),
            output_feature=(FeatureType.DATA_TIMELESS, self.config.output_feature_name),
            output_dtype=self.config.dtype,
            clip_predictions=self.config.clip_predictions,
        )
        return EONode(prediction_task, inputs=[previous_node])


class ClassificationPredictionPipeline(BasePredictionPipeline):
    class Schema(BasePredictionPipeline.Schema):
        output_feature_name: str
        output_probability_feature_name: Optional[str]

        model_filename: str = Field(description="A filename of a classification model to be used for prediction.")
        label_encoder_filename: Optional[str] = Field(
            description=(
                "Whether the predictions need to be decoded. The label encoder should be in the same model folder."
            )
        )

    config: Schema

    def _get_output_features(self) -> list[Feature]:
        features = [(FeatureType.MASK_TIMELESS, self.config.output_feature_name)]
        if self.config.output_probability_feature_name:
            features.append((FeatureType.DATA_TIMELESS, self.config.output_probability_feature_name))
        return features

    def _get_prediction_node(self, previous_node: EONode) -> EONode:
        model_path = fs.path.join(self.storage.get_folder(self.config.model_folder_key), self.config.model_filename)
        prediction_task = ClassificationPredictionTask(
            model_path=model_path,
            filesystem=self.storage.filesystem,
            input_features=self.config.input_features,
            mask_feature=_optional_typed_feature(FeatureType.MASK_TIMELESS, self.config.prediction_mask_feature_name),
            output_feature=(FeatureType.MASK_TIMELESS, self.config.output_feature_name),
            output_probability_feature=_optional_typed_feature(
                FeatureType.DATA_TIMELESS, self.config.output_probability_feature_name
            ),
            output_dtype=self.config.dtype,
            label_encoder_filename=self.config.label_encoder_filename,
        )
        return EONode(prediction_task, inputs=[previous_node])


def _optional_typed_feature(ftype: FeatureType, fname: str | None) -> Feature | None:
    """Constructs a typed feature if possible"""
    return (ftype, fname) if fname is not None else None
