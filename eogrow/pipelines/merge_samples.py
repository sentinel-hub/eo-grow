"""
Module implementing merge samples
"""
import logging
from typing import List, Literal, Optional, Tuple, cast

import fs
import numpy as np
from pydantic import Field

from eolearn.core import EOPatch, EOWorkflow, FeatureType, LoadTask, OutputTask, linearly_connect_tasks
from eolearn.core.utils.fs import get_full_path

from ..core.pipeline import Pipeline
from ..utils.types import Feature, FeatureSpec

LOGGER = logging.getLogger(__name__)


class MergeSamplesPipeline(Pipeline):
    """Pipeline to merge sampled data into joined numpy arrays"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder for the merge samples."
        )
        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the merge samples pipeline."
        )
        features_to_merge: List[Feature] = Field(
            description="Dictionary of all features for which samples are to be merged."
        )
        include_timestamp: bool = Field(False, description="Whether to also prepare an array of merged timestamps.")
        id_filename: Optional[str] = Field(description="Filename of array holding patch id of concatenated features")
        suffix: str = Field("", description="String to append to array filenames")
        workers: int = Field(1, description="Number of threads used to load data from EOPatches in parallel.")
        use_ray: Literal[False] = Field(False, description="Pipeline does not parallelize properly.")
        skip_existing: Literal[False] = False

    config: Schema

    _OUTPUT_NAME = "features-to-merge"

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Procedure which merges data from EOPatches into ML-ready numpy arrays"""
        workflow = self.build_workflow()
        exec_args = self.get_execution_arguments(workflow)

        # It doesn't make sense to parallelize loading over a cluster, but it would # make sense to parallelize over
        # features that have to be concatenated or, if we would concatenate into multiple files, parallelize creating
        # batches of features
        successful, failed, results = self.run_execution(workflow, exec_args, multiprocess=False)

        result_patches = [cast(EOPatch, result.outputs.get(self._OUTPUT_NAME)) for result in results]

        self.merge_and_save_features(result_patches, patch_names=successful)

        return successful, failed

    def build_workflow(self) -> EOWorkflow:
        """Creates a workflow that outputs the requested features"""
        features_to_load: List[FeatureSpec] = [FeatureType.TIMESTAMP] if self.config.include_timestamp else []
        features_to_load.extend(self.config.features_to_merge)
        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key),
            filesystem=self.storage.filesystem,
            features=features_to_load,
        )
        output_task = OutputTask(name=self._OUTPUT_NAME)
        return EOWorkflow(linearly_connect_tasks(load_task, output_task))

    def merge_and_save_features(self, patches: List[EOPatch], patch_names: List[str]) -> None:
        """Merges features from EOPatches and saves data"""
        patch_sample_nums = None

        for feature in self.config.features_to_merge:
            LOGGER.info("Started merging feature %s", feature)
            arrays = [self._collect_and_remove_feature(patch, feature) for patch in patches]
            _, feature_name = feature

            if patch_sample_nums is None:
                patch_sample_nums = [array.shape[0] for array in arrays]

            merged_array: np.ndarray = np.concatenate(arrays, axis=0)
            del arrays

            self._save_array(merged_array, feature_name)
            del merged_array

        if patch_sample_nums is None:
            raise ValueError("Need at least one feature to merge.")

        if self.config.include_timestamp:
            arrays = []
            for patch, sample_num in zip(patches, patch_sample_nums):
                arrays.append(np.tile(np.array(patch.timestamp), (sample_num, 1)))
                patch.timestamp = []

            self._save_array(np.concatenate(arrays, axis=0), "TIMESTAMPS")

        if self.config.id_filename:
            LOGGER.info("Started merging EOPatch ids")
            patch_ids = self.eopatch_manager.get_id_list_from_eopatch_list(patch_names)
            patch_id_arrays = [
                np.ones(sample_num, dtype=np.uint32) * patch_id
                for sample_num, patch_id in zip(patch_sample_nums, patch_ids)
            ]

            merged_patch_ids: np.ndarray = np.concatenate(patch_id_arrays, axis=0)
            self._save_array(merged_patch_ids, self.config.id_filename)

    @staticmethod
    def _collect_and_remove_feature(patch: EOPatch, feature: Feature) -> np.ndarray:
        """Collects a feature from an EOPatch and removes it from EOPatch to conserve overall memory"""
        feature_array = patch[feature]
        feature_type, _ = feature

        if feature_type is FeatureType.TIMESTAMP:
            patch.timestamp = []
            return np.array(feature_array)

        del patch[feature]

        axis = feature_type.ndim() - 2  # type: ignore[operator]
        feature_array = np.squeeze(feature_array, axis=axis)

        if feature_type in [FeatureType.DATA, FeatureType.MASK]:
            feature_array = np.moveaxis(feature_array, 0, 1)

        if feature_array.dtype == np.float64:
            feature_array = feature_array.astype(np.float32)

        return feature_array

    def _save_array(self, array: np.ndarray, name: str) -> None:
        """Saves an array to the storage with training data using the name of the array"""
        save_folder = self.storage.get_folder(self.config.output_folder_key)
        if self.config.suffix:
            name = f"{name}_{self.config.suffix}"
        path = fs.path.combine(save_folder, f"{name}.npy")

        with self.storage.filesystem.openbin(path, "w") as file_handle:
            np.save(file_handle, array)

        LOGGER.info("Saved concatenated array to %s", get_full_path(self.storage.filesystem, path))
