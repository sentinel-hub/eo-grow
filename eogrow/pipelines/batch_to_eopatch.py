"""Pipeline for conversion of batch results to EOPatches."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from pydantic import Field, validator

from eolearn.core import (
    CreateEOPatchTask,
    EONode,
    EOWorkflow,
    FeatureType,
    MergeEOPatchesTask,
    MergeFeatureTask,
    OverwritePermission,
    RemoveFeatureTask,
    RenameFeatureTask,
    SaveTask,
)
from eolearn.core.types import Feature
from eolearn.io import ImportFromTiffTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..tasks.batch_to_eopatch import DeleteFilesTask, FixImportedTimeDependentFeatureTask, LoadUserDataTask
from ..tasks.common import LinearFunctionTask
from ..types import ExecKwargs, PatchList, RawSchemaDict
from ..utils.filter import get_patches_with_missing_features
from ..utils.validators import ensure_storage_key_presence, optional_field_validator, parse_dtype


class FeatureMappingSchema(BaseSchema):
    """Defines a mapping between 1 or more batch outputs into an EOPatch feature"""

    batch_files: List[str] = Field(
        description=(
            "A list of files that will be converted into an EOPatch feature. If you specify multiple tiff "
            "files they will be concatenated together over the bands dimension in the specified order."
        ),
    )
    feature: Feature
    multiply_factor: float = Field(1, description="Factor used to multiply feature values with.")
    dtype: Optional[np.dtype] = Field(description="Dtype of the output feature.")
    _parse_dtype = optional_field_validator("dtype", parse_dtype, pre=True)


class BatchToEOPatchPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(description="Storage manager key pointing to the path with Batch results")
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        output_folder_key: str = Field(description="Storage manager key pointing to where the EOPatches are saved")
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        userdata_feature_name: Optional[str] = Field(
            description="A name of META_INFO feature in which userdata.json would be stored."
        )
        userdata_timestamp_reader: Optional[str] = Field(
            description=(
                "Either an import path to a utility function or a Python code describing how to read "
                "dates from userdata dictionary."
            ),
            example="\"[info['date'] for info in json.loads(userdata['metadata'])]\"",
        )

        mapping: List[FeatureMappingSchema] = Field(
            description="A list of mapping from batch files into EOPatch features."
        )

        @validator("mapping")
        def check_nonempty_input(cls, value: list, values: RawSchemaDict) -> list:
            if not value:
                params = "userdata_feature_name", "userdata_timestamp_reader"
                assert any(
                    values.get(param) is not None for param in params
                ), "At least one of `userdata_feature_name`, `userdata_timestamp_reader`, or `mapping` has to be set."
            return value

        remove_batch_data: bool = Field(True, description="Remove the raw batch data after the conversion is complete")

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        """Additionally sets some basic parameters calculated from config parameters"""
        super().__init__(*args, **kwargs)

        self._input_folder = self.storage.get_folder(self.config.input_folder_key)
        self._has_userdata = self.config.userdata_feature_name or self.config.userdata_timestamp_reader
        self._all_batch_files = self._get_all_batch_files()

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of specified output features"""
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
            check_timestamps=self.config.userdata_timestamp_reader is not None,
        )

    def _get_output_features(self) -> list[Feature]:
        """Lists all features that the pipeline outputs."""
        features = [feature_mapping.feature for feature_mapping in self.config.mapping]

        if self.config.userdata_feature_name:
            features.append((FeatureType.META_INFO, self.config.userdata_feature_name))

        return features

    def build_workflow(self) -> EOWorkflow:
        """Builds the workflow"""
        metadata_node = EONode(CreateEOPatchTask(), name="Establish BBox")
        if self._has_userdata:
            metadata_node = EONode(
                LoadUserDataTask(
                    path=self._input_folder,
                    filesystem=self.storage.filesystem,
                    userdata_feature_name=self.config.userdata_feature_name,
                    userdata_timestamp_reader=self.config.userdata_timestamp_reader,
                ),
                inputs=[metadata_node],
            )

        mapping_nodes = [
            self._get_tiff_mapping_node(feature_mapping, metadata_node) for feature_mapping in self.config.mapping
        ]

        last_node = metadata_node
        if len(mapping_nodes) == 1:
            last_node = mapping_nodes[0]
        elif len(mapping_nodes) > 1:
            last_node = EONode(MergeEOPatchesTask(), inputs=mapping_nodes)

        processing_node = self.get_processing_node(last_node)

        save_task = SaveTask(
            path=self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            features=self._get_output_features(),
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
        )
        save_node = EONode(save_task, inputs=([processing_node] if processing_node else []))

        cleanup_node = None
        if self.config.remove_batch_data:
            delete_task = DeleteFilesTask(
                path=self._input_folder,
                filesystem=self.storage.filesystem,
                filenames=self._all_batch_files,
            )
            cleanup_node = EONode(delete_task, inputs=[save_node], name="Delete batch data")

        return EOWorkflow.from_endnodes(cleanup_node or save_node)

    def _get_tiff_mapping_node(self, mapping: FeatureMappingSchema, previous_node: EONode | None) -> EONode:
        """Prepares tasks and dependencies that convert tiff files into an EOPatch feature"""
        if not all(batch_file.endswith(".tif") for batch_file in mapping.batch_files):
            raise ValueError(f"All batch files should end with .tif but found {mapping.batch_files}")

        feature_type, feature_name = mapping.feature
        if not (feature_type.is_image()):
            raise ValueError(f"Tiffs can only be read into spatial raster feature types, but {feature_type} was given.")

        tmp_features = []
        end_nodes = []
        for batch_file in mapping.batch_files:
            feature = feature_type, batch_file.replace(".tif", "_tmp")
            tmp_features.append(feature)

            tmp_timeless_feature = (
                FeatureType.MASK_TIMELESS if feature_type.is_discrete() else FeatureType.DATA_TIMELESS
            ), feature[1]

            import_task = ImportFromTiffTask(
                tmp_timeless_feature, self._input_folder, filesystem=self.storage.filesystem
            )
            # Filename is written into the dependency name to be used later for execution arguments:
            import_node = EONode(
                import_task,
                inputs=[previous_node] if previous_node else [],
                name=f"{batch_file} import",
            )

            if feature_type.is_temporal():
                fix_task = FixImportedTimeDependentFeatureTask(tmp_timeless_feature, feature)
                end_nodes.append(EONode(fix_task, inputs=[import_node]))
            else:
                end_nodes.append(import_node)

        previous_node = EONode(MergeEOPatchesTask(), inputs=end_nodes) if len(end_nodes) > 1 else end_nodes[0]

        final_feature = feature_type, feature_name
        end_node = self._get_feature_merge_node(previous_node, tmp_features, final_feature)

        if mapping.multiply_factor != 1 or mapping.dtype is not None:
            multiply_task = LinearFunctionTask(final_feature, slope=mapping.multiply_factor, dtype=mapping.dtype)
            end_node = EONode(multiply_task, inputs=[end_node])

        return end_node

    @staticmethod
    def _get_feature_merge_node(
        previous_node: EONode, input_features: list[Feature], output_feature: Feature
    ) -> EONode:
        """Merges input features into a single output feature and removes input features. In case there is a single
        input feature this method just renames it into the output feature. This way it avoids memory duplication that
        otherwise happens in `MergeFeatureTask`."""
        if len(input_features) == 1:
            feature_type, input_feature_name = input_features[0]
            _, output_feature_name = output_feature

            rename_task = RenameFeatureTask([(feature_type, input_feature_name, output_feature_name)])
            return EONode(rename_task, inputs=[previous_node])

        merge_feature_task = MergeFeatureTask(input_features=input_features, output_feature=output_feature)
        merge_node = EONode(merge_feature_task, inputs=[previous_node])

        remove_task = RemoveFeatureTask(input_features)
        return EONode(remove_task, inputs=[merge_node])

    def get_processing_node(self, previous_node: EONode) -> EONode:
        """This method can be overwritten to add more tasks that process loaded data before saving it."""
        return previous_node

    def get_execution_arguments(self, workflow: EOWorkflow, patch_list: PatchList) -> ExecKwargs:
        """Prepare execution arguments per each EOPatch"""
        exec_args = super().get_execution_arguments(workflow, patch_list)

        nodes = workflow.get_nodes()

        for patch_name, patch_args in exec_args.items():
            for node in nodes:
                if isinstance(node.task, ImportFromTiffTask):
                    if node.name is None:
                        raise RuntimeError("One of the ImportFromTiffTask nodes has not been tagged with filename.")
                    filename = node.name.split()[0]
                    path = f"{patch_name}/{filename}"
                    patch_args[node] = dict(filename=path)

                if isinstance(node.task, (DeleteFilesTask, LoadUserDataTask)):
                    patch_args[node] = dict(folder=patch_name)

        return exec_args

    def _get_all_batch_files(self) -> list[str]:
        """Provides a list of batch files used in this pipeline"""
        files = [file for feature_mapping in self.config.mapping for file in feature_mapping.batch_files]

        if self._has_userdata:
            files.append("userdata.json")

        return list(set(files))
