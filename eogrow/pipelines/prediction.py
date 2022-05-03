"""
Module implementing prediction pipeline
"""
import abc
from typing import List, Optional, Tuple

from pydantic import Field, validator

from eolearn.core import EONode, EOWorkflow, FeatureType, LoadTask, MergeEOPatchesTask, OverwritePermission, SaveTask

from ..core.pipeline import Pipeline
from ..tasks.prediction import ClassificationPredictionTask, RegressionPredictionTask
from ..utils.filter import get_patches_with_missing_features
from ..utils.types import Feature, FeatureSpec


class BasePredictionPipeline(Pipeline, metaclass=abc.ABCMeta):
    """Pipeline to load a model and run prediction on EOPatches"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder of the model input data."
        )
        input_features: List[Feature] = Field(
            description=(
                "List of features of form `[(feature_type, feature_name)]` specifying which features are model input in"
                " the correct order"
            )
        )

        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the prediction pipeline."
        )
        dtype: Optional[str] = Field(
            description="Casts the result to desired type. Uses predictor output type by default."
        )

        prediction_mask_folder_key: Optional[str]
        prediction_mask_feature_name: Optional[str] = Field(
            description="Name of `MASK_TIMELESS` feature which defines which areas will be predicted"
        )

        @validator("prediction_mask_feature_name")
        def validate_filename(cls, v, values):  # type: ignore
            if values["prediction_mask_folder_key"] is None:
                assert v is None, "Both `prediction_mask_folder_key` and `prediction_mask_feature_name` must be given."
            return v

        model_folder_key: str = Field(
            description="The storage manager key pointing to the folder of the model used in the prediction pipeline."
        )
        compress_level: int = Field(1, description="Level of compression used in saving EOPatches")

    config: Schema

    @abc.abstractmethod
    def _get_output_features(self) -> List[FeatureSpec]:
        """Lists all features that are to be saved upon the pipeline completion"""

    @property
    def _is_mp_lock_needed(self) -> bool:
        """If a multiprocessing lock is needed when executing"""
        return not self.config.use_ray and self.config.workers > 1

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """EOPatches are filtered according to existence of specified output features"""

        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
        )

        return filtered_patch_list

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
                self.storage.get_folder(self.config.input_folder_key, full_path=True),
                features=[FeatureType.BBOX, FeatureType.TIMESTAMP, *self.config.input_features],
                config=self.sh_config,
            )
        )

        if not self.config.prediction_mask_folder_key:
            return features_load_node

        mask_load_node = EONode(
            LoadTask(
                self.storage.get_folder(self.config.prediction_mask_folder_key, full_path=True),
                features=[(FeatureType.MASK_TIMELESS, self.config.prediction_mask_feature_name)],
                config=self.sh_config,
            )
        )

        return EONode(MergeEOPatchesTask(), inputs=[features_load_node, mask_load_node])

    @abc.abstractmethod
    def _get_prediction_node(self, previous_node: EONode) -> EONode:
        """Returns nodes for using the model on the prepared data and the endpoints of the dependencies"""

    def _get_saving_node(self, previous_node: EONode) -> EONode:
        """Returns nodes for finalizing and saving features"""
        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key, full_path=True),
            features=self._get_output_features(),
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
            compress_level=self.config.compress_level,
        )

        return EONode(save_task, inputs=[previous_node])


class RegressionPredictionPipeline(BasePredictionPipeline):
    class Schema(BasePredictionPipeline.Schema):
        output_feature_name: str
        model_filename: str = Field(description="A filename of a regression model to be used for prediction.")
        clip_predictions: Optional[Tuple[float, float]] = Field(
            description="Whether to clip values of predictions to specified interval"
        )

    config: Schema

    def _get_output_features(self) -> List[FeatureSpec]:
        return [FeatureType.BBOX, (FeatureType.DATA_TIMELESS, self.config.output_feature_name)]

    def _get_prediction_node(self, previous_node: EONode) -> EONode:
        prediction_task = RegressionPredictionTask(
            model_folder=self.storage.get_folder(self.config.model_folder_key, full_path=True),
            model_filename=self.config.model_filename,
            input_features=self.config.input_features,
            mask_feature=_optional_typed_feature(FeatureType.MASK_TIMELESS, self.config.prediction_mask_feature_name),
            output_feature=(FeatureType.DATA_TIMELESS, self.config.output_feature_name),
            output_dtype=self.config.dtype,
            mp_lock=self._is_mp_lock_needed,
            sh_config=self.sh_config,
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

    def _get_output_features(self) -> List[FeatureSpec]:
        features: List[FeatureSpec] = [FeatureType.BBOX, (FeatureType.MASK_TIMELESS, self.config.output_feature_name)]
        if self.config.output_probability_feature_name:
            features.append((FeatureType.DATA_TIMELESS, self.config.output_probability_feature_name))
        return features

    def _get_prediction_node(self, previous_node: EONode) -> EONode:
        prediction_task = ClassificationPredictionTask(
            model_folder=self.storage.get_folder(self.config.model_folder_key, full_path=True),
            model_filename=self.config.model_filename,
            input_features=self.config.input_features,
            mask_feature=_optional_typed_feature(FeatureType.MASK_TIMELESS, self.config.prediction_mask_feature_name),
            output_feature=(FeatureType.MASK_TIMELESS, self.config.output_feature_name),
            output_probability_feature=_optional_typed_feature(
                FeatureType.DATA_TIMELESS, self.config.output_probability_feature_name
            ),
            output_dtype=self.config.dtype,
            mp_lock=self._is_mp_lock_needed,
            sh_config=self.sh_config,
            label_encoder_filename=self.config.label_encoder_filename,
        )
        return EONode(prediction_task, inputs=[previous_node])


def _optional_typed_feature(ftype: FeatureType, fname: Optional[str]) -> Optional[Feature]:
    """Constructs a typed feature if possible"""
    return (ftype, fname) if fname is not None else None
