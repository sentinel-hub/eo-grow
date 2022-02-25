"""
Conversion of batch results to EOPatches
"""
from typing import Any, Dict, List, Optional

import fs
from pydantic import Field

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


class FeatureMappingSchema(BaseSchema):
    """Defines a mapping between 1 or more batch outputs into an EOPatch feature"""

    batch_files: List[str] = Field(
        description=(
            "A list of files that will be converted into an EOPatch feature. If you specify multiple tiff "
            "files they will be concatenated together over the bands dimension in the specified order."
        ),
    )
    feature_type: FeatureType
    feature_name: str
    multiply_factor: Optional[float] = Field(description="Factor used to multiply feature values with.")
    dtype: Optional[str] = Field(
        "float32",
        description=(
            "Dtype of the output feature. Only taken into account if `multiply_factor` is used."
            "Default set to `float32`."
        ),
    )


class BatchToEOPatchPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        folder_key: str = Field(description="Storage manager key pointing to the path with Batch results")
        mapping: List[FeatureMappingSchema] = Field(
            description="A list of mapping from batch files into EOPatch features."
        )
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

    def __init__(self, *args: Any, **kwargs: Any):
        """Additionally sets some basic parameters calculated from config parameters"""
        super().__init__(*args, **kwargs)

        self._processing_folder = self.storage.get_folder(self.config.folder_key, full_path=True)
        self._relative_processing_folder = self.storage.get_folder(self.config.folder_key)
        self._all_batch_files = self._get_all_batch_files()

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """Checks the content of EOPatches in the project folder. If some files, produced by Sentinel Hub Batch, still
        exist this means that EOPatch hasn't been yet successfully processed by this pipeline. That is because this
        pipeline deletes batch files at the end. Such patches will remain after this filtering.
        """
        processed_eopatches = set()
        for eopatch_name in patch_list:
            eopatch_folder = fs.path.combine(self._relative_processing_folder, eopatch_name)
            eopatch_files = set(self.storage.filesystem.listdir(eopatch_folder))

            if not any(batch_file in eopatch_files for batch_file in self._all_batch_files):
                processed_eopatches.add(eopatch_name)

        return [eopatch for eopatch in patch_list if eopatch not in processed_eopatches]

    def build_workflow(self) -> EOWorkflow:
        """Builds the workflow"""
        userdata_node = EONode(
            LoadUserDataTask(
                path=self._processing_folder,
                userdata_feature_name=self.config.userdata_feature_name,
                userdata_timestamp_reader=self.config.userdata_timestamp_reader,
                config=self.sh_config,
            )
        )

        mapping_nodes = [
            self._get_tiff_mapping_node(feature_mapping, userdata_node) for feature_mapping in self.config.mapping
        ]

        last_node = mapping_nodes[0] if len(mapping_nodes) == 1 else userdata_node
        if len(mapping_nodes) > 1:
            last_node = EONode(MergeEOPatchesTask(), inputs=mapping_nodes)

        processing_node = self.get_processing_node(last_node)

        save_task = SaveTask(
            path=self._processing_folder,
            compress_level=1,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
        )
        save_node = EONode(save_task, inputs=[processing_node])

        delete_task = DeleteFilesTask(
            path=self._processing_folder, filenames=self._all_batch_files, config=self.sh_config
        )
        cleanup_node = EONode(delete_task, inputs=[save_node], name="Delete batch data")

        return EOWorkflow.from_endnodes(cleanup_node)

    def _get_tiff_mapping_node(self, mapping: FeatureMappingSchema, previous_node: EONode) -> EONode:
        """Prepares tasks and dependencies that convert tiff files into an EOPatch feature"""
        if not all(batch_file.endswith(".tif") for batch_file in mapping.batch_files):
            raise ValueError(f"All batch files should end with .tif but found {mapping.batch_files}")

        feature_type = mapping.feature_type
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

            import_task = ImportFromTiffTask(
                tmp_timeless_feature, folder=self._processing_folder, config=self.sh_config
            )
            # Filename is written into the dependency name to be used later for execution arguments:
            import_node = EONode(import_task, inputs=[previous_node], name=f"{batch_file} import")

            if feature_type.is_temporal():
                fix_task = FixImportedTimeDependentFeatureTask(tmp_timeless_feature, feature)
                end_nodes.append(EONode(fix_task, inputs=[import_node]))
            else:
                end_nodes.append(import_node)

        if len(end_nodes) == 1:
            previous_node = end_nodes[0]
        if len(end_nodes) > 1:
            previous_node = EONode(MergeEOPatchesTask(), inputs=end_nodes)

        final_feature = feature_type, mapping.feature_name
        merge_feature_task = MergeFeatureTask(input_features=tmp_features, output_feature=final_feature)
        merge_node = EONode(merge_feature_task, inputs=[previous_node])

        end_node = EONode(RemoveFeatureTask(tmp_features), inputs=[merge_node])

        if mapping.multiply_factor is not None:
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
                    filename = node.name.split()[0]
                    path = f"{name}/{filename}"
                    single_exec_dict[node] = dict(filename=path)

                if isinstance(node.task, (DeleteFilesTask, LoadUserDataTask)):
                    single_exec_dict[node] = dict(folder=name)

        return exec_args

    def _get_all_batch_files(self) -> List[str]:
        """Provides a list of batch files used in this pipeline"""
        files = [file for feature_mapping in self.config.mapping for file in feature_mapping.batch_files]

        if self.config.userdata_feature_name or self.config.userdata_timestamps_path:
            files.append("userdata.json")

        return list(set(files))
