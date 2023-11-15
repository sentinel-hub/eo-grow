"""Implements a pipeline for merging sampled features into numpy arrays fit for training models."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, cast

import fs
import numpy as np
from pydantic import Field

from eolearn.core import EOExecutor, EOPatch, EOWorkflow, FeatureType, LoadTask, OutputTask, linearly_connect_tasks
from eolearn.core.types import Feature
from eolearn.core.utils.fs import get_full_path

from ..core.logging import EOExecutionFilter, EOExecutionHandler
from ..core.pipeline import Pipeline
from ..utils.validators import ensure_storage_key_presence

LOGGER = logging.getLogger(__name__)


class MergeSamplesPipeline(Pipeline):
    """Pipeline to merge sampled data into joined numpy arrays"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder for the merge samples."
        )
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the merge samples pipeline."
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        features_to_merge: List[Feature] = Field(
            description="Dictionary of all features for which samples are to be merged."
        )
        id_filename: Optional[str] = Field(
            description=(
                "Filename of array holding patch ID of concatenated features. The patch ID is the index of the patch in"
                " the final patch list, any filtration of the patch list will impact the results."
            )
        )
        suffix: str = Field("", description="String to append to array filenames")
        num_threads: int = Field(1, description="Number of threads used to load data from EOPatches in parallel.")
        skip_existing: Literal[False] = False

    config: Schema

    _OUTPUT_NAME = "features-to-merge"

    def run_procedure(self) -> tuple[list[str], list[str]]:
        """Procedure which merges data from EOPatches into ML-ready numpy arrays"""
        workflow = self.build_workflow()
        patch_list = self.get_patch_list()
        exec_args = self.get_execution_arguments(workflow, patch_list)

        # It doesn't make sense to parallelize loading over a cluster, but it would # make sense to parallelize over
        # features that have to be concatenated or, if we would concatenate into multiple files, parallelize creating
        # batches of features
        LOGGER.info("Starting processing for %d EOPatches", len(exec_args))

        # Unpacking manually to ensure order matches
        list_of_kwargs, execution_names = [], []
        for exec_name, exec_kwargs in exec_args.items():
            list_of_kwargs.append(exec_kwargs)
            execution_names.append(exec_name)

        executor = EOExecutor(
            workflow,
            list_of_kwargs,
            execution_names=execution_names,
            save_logs=self.logging_manager.config.save_logs,
            logs_folder=self.logging_manager.get_pipeline_logs_folder(self.current_execution_name),
            filesystem=self.storage.filesystem,
            logs_filter=EOExecutionFilter(ignore_packages=self.logging_manager.config.eoexecution_ignore_packages),
            logs_handler_factory=EOExecutionHandler,
            raise_on_temporal_mismatch=self.config.raise_on_temporal_mismatch,
        )
        execution_results = executor.run(multiprocess=True, workers=self.config.num_threads)

        successful = [execution_names[idx] for idx in executor.get_successful_executions()]
        failed = [execution_names[idx] for idx in executor.get_failed_executions()]
        LOGGER.info("EOExecutor finished with %d / %d success rate", len(successful), len(successful) + len(failed))

        if self.logging_manager.config.save_logs:
            executor.make_report(include_logs=self.logging_manager.config.include_logs_to_report)
            LOGGER.info("Saved EOExecution report to %s", executor.get_report_path(full_path=True))

        result_patches = [cast(EOPatch, result.outputs.get(self._OUTPUT_NAME)) for result in execution_results]

        self.merge_and_save_features(result_patches)

        return successful, failed

    def build_workflow(self) -> EOWorkflow:
        """Creates a workflow that outputs the requested features"""
        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key),
            filesystem=self.storage.filesystem,
            features=self.config.features_to_merge,
        )
        output_task = OutputTask(name=self._OUTPUT_NAME)
        return EOWorkflow(linearly_connect_tasks(load_task, output_task))

    def merge_and_save_features(self, patches: list[EOPatch]) -> None:
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

        if self.config.id_filename:
            LOGGER.info("Started merging EOPatch ids")
            patch_id_arrays = [
                np.ones(sample_num, dtype=np.uint32) * patch_id for patch_id, sample_num in enumerate(patch_sample_nums)
            ]

            merged_patch_ids: np.ndarray = np.concatenate(patch_id_arrays, axis=0)
            self._save_array(merged_patch_ids, self.config.id_filename)

    @staticmethod
    def _collect_and_remove_feature(patch: EOPatch, feature: Feature) -> np.ndarray:
        """Collects a feature from an EOPatch and removes it from EOPatch to conserve overall memory"""
        feature_array = patch[feature]
        feature_type, _ = feature

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
