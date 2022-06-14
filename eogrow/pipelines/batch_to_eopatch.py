"""
Conversion of batch results to EOPatches
"""
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field, root_validator

from eolearn.core import (
    EONode,
    EOWorkflow,
    FeatureType,
    MergeEOPatchesTask,
    MergeFeatureTask,
    OverwritePermission,
    RemoveFeatureTask,
    SaveTask,
)
from eolearn.features import LinearFunctionTask
from eolearn.io import ImportFromTiffTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..tasks.batch_to_eopatch import DeleteFilesTask, FixImportedTimeDependentFeatureTask, LoadUserDataTask
from ..utils.filter import get_patches_with_missing_features
from ..utils.types import Feature, FeatureSpec
from ..utils.validators import field_validator, optional_field_validator, parse_dtype


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
    save_early: bool = Field(
        True,
        description=(
            "The pipeline allows that custom processing steps are added at the end. If this parameter is set to True"
            " then the feature will be saved and removed from memory immediately after being loaded from tiffs. This"
            " makes the pipeline more memory efficient. If set to False it will keep the feature in memory until the"
            " end so that it can be used for processing steps."
        ),
    )


def _sort_feature_mappings(mapping: List[FeatureMappingSchema]) -> List[FeatureMappingSchema]:
    """Sorts feature mappings by putting the ones that will be saved early at the beginning. This optimizes the memory
    usage."""
    return sorted(mapping, key=lambda feature_mapping: not feature_mapping.save_early)


class BatchToEOPatchPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(description="Storage manager key pointing to the path with Batch results")
        output_folder_key: str = Field(description="Storage manager key pointing to where the eopatches are saved")

        mapping: List[FeatureMappingSchema] = Field(
            description="A list of mapping from batch files into EOPatch features."
        )
        _sort_mapping = field_validator("mapping", _sort_feature_mappings)

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
        remove_batch_data: bool = Field(
            True, description="Whether to remove the raw batch data after the conversion is complete"
        )

        @root_validator
        def check_something_is_converted(cls, values):  # type: ignore[no-untyped-def]
            """Check that the pipeline has something to do."""
            params = "userdata_feature_name", "userdata_timestamp_reader", "mapping"
            assert any(
                values.get(param) is not None for param in params
            ), "At least one of `userdata_feature_name`, `userdata_timestamp_reader`, or `mapping` has to be set."
            return values

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        """Additionally sets some basic parameters calculated from config parameters"""
        super().__init__(*args, **kwargs)

        self._input_folder = self.storage.get_folder(self.config.input_folder_key, full_path=True)
        self._has_userdata = self.config.userdata_feature_name or self.config.userdata_timestamp_reader
        self._all_batch_files = self._get_all_batch_files()

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """EOPatches are filtered according to existence of specified output features"""

        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
        )

        return filtered_patch_list

    def _get_output_features(self) -> List[FeatureSpec]:
        """Lists all features that the pipeline outputs."""
        features: List[FeatureSpec] = [FeatureType.BBOX]
        features.extend(feature_mapping.feature for feature_mapping in self.config.mapping)

        if self.config.userdata_feature_name:
            features.append(FeatureType.META_INFO)

        if self.config.userdata_timestamp_reader:
            features.append(FeatureType.TIMESTAMP)

        return features

    def _get_final_output_features(self) -> List[FeatureSpec]:
        """Output features that will be saved in the final saving step and not sooner."""
        output_features = self._get_output_features()
        features_saved_early = {
            feature_mapping.feature for feature_mapping in self.config.mapping if feature_mapping.save_early
        }
        return [feature for feature in output_features if feature not in features_saved_early]

    def build_workflow(self) -> EOWorkflow:
        """Builds the workflow"""
        previous_node = None
        if self._has_userdata:
            previous_node = EONode(
                LoadUserDataTask(
                    path=self._input_folder,
                    userdata_feature_name=self.config.userdata_feature_name,
                    userdata_timestamp_reader=self.config.userdata_timestamp_reader,
                    config=self.sh_config,
                )
            )

        for feature_mapping in self.config.mapping:
            feature = feature_mapping.feature
            mapping_node = self._get_tiff_mapping_node(feature_mapping, previous_node)

            if feature_mapping.save_early:
                save_task = SaveTask(
                    path=self.storage.get_folder(self.config.output_folder_key, full_path=True),
                    features=[feature],
                    compress_level=1,
                    overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                    config=self.sh_config,
                )
                save_node = EONode(save_task, inputs=[mapping_node])

                _, f_name = feature
                previous_node = EONode(RemoveFeatureTask([feature]), inputs=[save_node], name=f"Remove {f_name}")
            else:
                previous_node = mapping_node

        if previous_node is None:
            raise ValueError(
                "At least one of `userdata_feature_name`, `userdata_timestamp_reader`, or `mapping` has to be set in"
                " the config. This should have been caught in the validation phase, please report issue."
            )

        processing_node = self.get_processing_node(previous_node)

        save_task = SaveTask(
            path=self.storage.get_folder(self.config.output_folder_key, full_path=True),
            features=self._get_final_output_features(),
            compress_level=1,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
        )
        save_node = EONode(save_task, inputs=([processing_node] if processing_node else []))

        cleanup_node = None
        if self.config.remove_batch_data:
            delete_task = DeleteFilesTask(
                path=self._input_folder, filenames=self._all_batch_files, config=self.sh_config
            )
            cleanup_node = EONode(delete_task, inputs=[save_node], name="Delete batch data")

        return EOWorkflow.from_endnodes(cleanup_node or save_node)

    def _get_tiff_mapping_node(self, mapping: FeatureMappingSchema, previous_node: Optional[EONode]) -> EONode:
        """Prepares tasks and dependencies that convert tiff files into an EOPatch feature"""
        if not all(batch_file.endswith(".tif") for batch_file in mapping.batch_files):
            raise ValueError(f"All batch files should end with .tif but found {mapping.batch_files}")

        feature_type, feature_name = mapping.feature
        if not (feature_type.is_spatial() and feature_type.is_raster()):
            raise ValueError(f"Tiffs can only be read into spatial raster feature types, but {feature_type} was given.")

        tmp_features = []
        end_nodes = []
        for batch_file in mapping.batch_files:
            feature = feature_type, batch_file.replace(".tif", "_tmp")
            tmp_features.append(feature)

            tmp_timeless_feature = (
                FeatureType.MASK_TIMELESS if feature_type.is_discrete() else FeatureType.DATA_TIMELESS
            ), feature[1]

            import_task = ImportFromTiffTask(tmp_timeless_feature, folder=self._input_folder, config=self.sh_config)
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
        merge_feature_task = MergeFeatureTask(input_features=tmp_features, output_feature=final_feature)
        merge_node = EONode(merge_feature_task, inputs=[previous_node])

        end_node = EONode(RemoveFeatureTask(tmp_features), inputs=[merge_node])

        if mapping.multiply_factor != 1 or mapping.dtype is not None:
            multiply_task = LinearFunctionTask(final_feature, slope=mapping.multiply_factor, dtype=mapping.dtype)
            end_node = EONode(multiply_task, inputs=[end_node])

        return end_node

    @staticmethod
    def get_processing_node(previous_node: EONode) -> EONode:
        """This method can be overwritten to add more tasks that process loaded data before saving it."""
        return previous_node

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        """Prepare execution arguments per each EOPatch"""
        exec_args = super().get_execution_arguments(workflow)

        nodes = workflow.get_nodes()

        for name, single_exec_dict in zip(self.patch_list, exec_args):
            for node in nodes:
                if isinstance(node.task, ImportFromTiffTask):
                    if node.name is None:
                        raise RuntimeError("One of the ImportFromTiffTask nodes has not been tagged with filename.")
                    filename = node.name.split()[0]
                    path = f"{name}/{filename}"
                    single_exec_dict[node] = dict(filename=path)

                if isinstance(node.task, (DeleteFilesTask, LoadUserDataTask)):
                    single_exec_dict[node] = dict(folder=name)

        return exec_args

    def _get_all_batch_files(self) -> List[str]:
        """Provides a list of batch files used in this pipeline"""
        files = [file for feature_mapping in self.config.mapping for file in feature_mapping.batch_files]

        if self._has_userdata:
            files.append("userdata.json")

        return list(set(files))
