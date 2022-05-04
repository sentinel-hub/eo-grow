"""
Module implementing sampling pipelines
"""
import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import Field, validator

from eolearn.core import EONode, EOWorkflow, FeatureType, LoadTask, MergeEOPatchesTask, OverwritePermission, SaveTask
from eolearn.geometry import MorphologicalOperations, MorphologicalStructFactory
from eolearn.ml_tools import BlockSamplingTask, FractionSamplingTask, GridSamplingTask

from ..core.pipeline import Pipeline
from ..tasks.common import ClassFilterTask
from ..utils.filter import get_patches_with_missing_features
from ..utils.types import Feature, FeatureSpec


class BaseSamplingPipeline(Pipeline, metaclass=abc.ABCMeta):
    """Pipeline to run sampling on EOPatches"""

    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(description="The storage manager key pointing to the pipeline output folder.")
        apply_to: Dict[str, Dict[FeatureType, List[str]]] = Field(
            description=(
                "A dictionary defining which features to sample, its structure is "
                "{folder_key: {feature_type: [feature_name]}}"
            ),
        )
        mask_of_samples_name: Optional[str] = Field(
            description=(
                "A name of a mask timeless output feature with information which pixels were sampled and how many times"
            )
        )
        sampled_suffix: Optional[str] = Field(
            description=(
                "If provided features are saved with a suffix, e.g. for suffix SAMPLED the sampled FEATURES are "
                "saved as FEATURES_SAMPLED."
            )
        )
        compress_level: int = Field(1, description="Level of compression used in saving eopatches")

    config: Schema

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """Filter output EOPatches that have already been processed"""
        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
        )

        return filtered_patch_list

    def build_workflow(self) -> EOWorkflow:
        """Creates workflow that is divided into the following sub-parts:

        1. loading data,
        2. preprocessing steps,
        3. sampling features
        4. saving results
        """
        loading_node = self._get_loading_node()
        preprocessing_node = self._get_preprocessing_node(loading_node)
        sampling_node = self._get_sampling_node(preprocessing_node)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key, full_path=True),
            features=self._get_output_features(),
            compress_level=self.config.compress_level,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
        )

        return EOWorkflow.from_endnodes(EONode(save_task, inputs=[sampling_node]))

    def _get_loading_node(self) -> EONode:
        """Prepares nodes for loading and joining EOPatches."""
        load_nodes = []

        for folder_name, features in self.config.apply_to.items():
            load_features: List[FeatureSpec] = []

            for feature_type_str, feature_names in features.items():
                feature_type = FeatureType(feature_type_str)

                if not feature_type.is_spatial():
                    raise ValueError(f"Only spatial features can be sampled, but found {feature_type}: {feature_names}")

                for feature_name in feature_names:
                    load_features.append((feature_type, feature_name))

            load_features.append(FeatureType.BBOX)
            if any(FeatureType(feature_type).is_temporal() for feature_type in features):
                load_features.append(FeatureType.TIMESTAMP)

            load_task = LoadTask(
                self.storage.get_folder(folder_name, full_path=True),
                lazy_loading=True,
                features=load_features,
                config=self.sh_config,
            )
            load_nodes.append(EONode(load_task, name=f"Load from {folder_name}"))

        return EONode(MergeEOPatchesTask(), inputs=load_nodes)

    def _get_preprocessing_node(self, previous_node: EONode) -> EONode:  # pylint: disable=no-self-use
        """The default implementation doesn't add any preprocessing steps"""
        return previous_node

    @abc.abstractmethod
    def _get_sampling_node(self, previous_node: EONode) -> EONode:
        """Method to prepare sampling nodes"""

    def _get_features_to_sample(self) -> List[Tuple[FeatureType, str, str]]:
        """Get a list of features that will be sampled, together with their new names"""
        features_to_sample = []
        for _, features in self.config.apply_to.items():
            for feature_type, feature_names in features.items():
                for feature_name in feature_names:
                    if self.config.sampled_suffix is None:
                        features_to_sample.append((feature_type, feature_name, feature_name))
                    else:
                        features_to_sample.append(
                            (feature_type, feature_name, f"{feature_name}_{self.config.sampled_suffix}")
                        )

        return features_to_sample

    def _get_mask_of_samples_feature(self) -> Optional[Feature]:
        """Provide a mask of samples feature"""
        if self.config.mask_of_samples_name:
            return FeatureType.MASK_TIMELESS, self.config.mask_of_samples_name
        return None

    def _get_output_features(self) -> List[FeatureSpec]:
        """Get a list of features that will be saved as an output of the pipeline"""
        output_features: List[FeatureSpec] = [FeatureType.BBOX]
        features_to_sample = self._get_features_to_sample()

        for feature_type, _, sampled_feature_name in features_to_sample:
            output_features.append((feature_type, sampled_feature_name))

        mask_of_samples_feature = self._get_mask_of_samples_feature()
        if mask_of_samples_feature:
            output_features.append(mask_of_samples_feature)

        if any(feature_type.is_temporal() for feature_type, _, _ in features_to_sample):
            output_features.append(FeatureType.TIMESTAMP)
        return output_features


class BaseRandomSamplingPipeline(BaseSamplingPipeline, metaclass=abc.ABCMeta):
    """A base class for all sampling pipeline that work on random selection of samples"""

    class Schema(BaseSamplingPipeline.Schema):
        seed: Optional[int] = Field(
            42, description="A random generator seed to be used in order to obtain the same results every pipeline run."
        )

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._sampling_node_uid: Optional[str] = None

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        """Extends the basic method for adding execution arguments by adding seed arguments a sampling task"""
        exec_args = super().get_execution_arguments(workflow)

        sampling_node = workflow.get_node_with_uid(self._sampling_node_uid)
        if sampling_node is None:
            return exec_args

        generator = np.random.default_rng(seed=self.config.seed)

        for arg_index in range(len(self.patch_list)):
            exec_args[arg_index][sampling_node] = dict(seed=generator.integers(low=0, high=2**32))

        return exec_args


class FractionSamplingPipeline(BaseRandomSamplingPipeline):
    """A pipeline to sample per-class with different distributions"""

    class Schema(BaseRandomSamplingPipeline.Schema):
        sampling_feature_name: str = Field(
            description="Name of MASK_TIMELESS feature to be used to create sample point"
        )
        erosion_dict: Optional[Dict[float, List[int]]] = Field(
            description="A dictionary specifying disc radius of erosion operation to be applied to a list of label IDs",
            example={2.5: [1, 3, 4], 1.0: [2]},
        )
        fraction_of_samples: Union[float, Dict[int, float]] = Field(
            description=(
                "A fraction or a dictionary of per-class fractions of valid pixels to sample from the the sampling "
                "feature."
            )
        )
        exclude_values: List[int] = Field(default_factory=list, description="Values to be excluded from sampling")

    config: Schema

    def _get_preprocessing_node(self, previous_node: EONode) -> EONode:
        """Preprocessing that applies erosion on sampling feature values"""
        if self.config.erosion_dict is None:
            return previous_node

        end_node = previous_node
        for radius, labels in self.config.erosion_dict.items():
            task = ClassFilterTask(
                (FeatureType.MASK_TIMELESS, self.config.sampling_feature_name),
                labels,
                MorphologicalOperations.EROSION,
                struct_elem=MorphologicalStructFactory.get_disk(radius),
            )
            end_node = EONode(task, inputs=[end_node])

        return end_node

    def _get_sampling_node(self, previous_node: EONode) -> EONode:
        """Prepare the sampling task"""
        task = FractionSamplingTask(
            features_to_sample=self._get_features_to_sample(),
            sampling_feature=(FeatureType.MASK_TIMELESS, self.config.sampling_feature_name),
            fraction=self.config.fraction_of_samples,
            exclude_values=self.config.exclude_values,
            mask_of_samples=self._get_mask_of_samples_feature(),
        )
        node = EONode(task, inputs=[previous_node])
        self._sampling_node_uid = node.uid

        return node


class BlockSamplingPipeline(BaseRandomSamplingPipeline):
    """A pipeline to randomly sample blocks"""

    class Schema(BaseRandomSamplingPipeline.Schema):
        sample_size: Tuple[int, int] = Field(description="A height and width of each block in pixels.")

        number_of_samples: Optional[int] = Field(
            description=(
                "A number of samples to be sampled. Exactly one of parameters fraction_of_samples and number_of_samples"
                " has to be given."
            )
        )
        fraction_of_samples: Optional[float] = Field(
            description=(
                "A percentage of samples to be sampled. Exactly one of parameters fraction_of_samples and "
                "number_of_samples has to be given."
            )
        )

        @validator("fraction_of_samples", always=True)
        def validate_sampling_params(cls, fraction, values):  # type: ignore
            """Makes sure only one of the sampling parameters has been given"""
            assert (fraction is None) != (
                values.get("number_of_samples") is None
            ), "Exactly one of the parameters fraction_of_samples and number_of_samples should be defined"
            return fraction

    config: Schema

    def _get_sampling_node(self, previous_node: EONode) -> EONode:
        """Prepare the sampling task"""
        task = BlockSamplingTask(
            features_to_sample=self._get_features_to_sample(),
            amount=self.config.fraction_of_samples or self.config.number_of_samples,
            sample_size=self.config.sample_size,
            mask_of_samples=self._get_mask_of_samples_feature(),
        )

        node = EONode(task, inputs=[previous_node])
        self._sampling_node_uid = node.uid

        return node


class GridSamplingPipeline(BaseSamplingPipeline):
    """A pipeline to sample blocks in a regular grid"""

    class Schema(BaseSamplingPipeline.Schema):
        sample_size: Tuple[int, int] = Field(description="A height and width of each block in pixels.")
        stride: Tuple[int, int] = Field(
            description=(
                "A tuple describing a distance between upper left corners of two consecutive sampled blocks. "
                "The first number is the vertical distance and the second number the horizontal distance. If stride in "
                "smaller than sample_size in any dimensions then sampled blocks will overlap."
            )
        )

    config: Schema

    def _get_sampling_node(self, previous_node: EONode) -> EONode:
        """Prepare the sampling task"""
        task = GridSamplingTask(
            features_to_sample=self._get_features_to_sample(),
            sample_size=self.config.sample_size,
            stride=self.config.stride,
            mask_of_samples=self._get_mask_of_samples_feature(),
        )

        return EONode(task, inputs=[previous_node])
