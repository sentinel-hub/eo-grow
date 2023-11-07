"""Implements a pipeline for importing vector data from a file."""

from __future__ import annotations

import fs
from pydantic import Field

from eolearn.core import EONode, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.core.types import Feature
from eolearn.io import VectorImportTask

from ..core.pipeline import Pipeline
from ..types import ExecKwargs, PatchList
from ..utils.validators import ensure_storage_key_presence, field_validator, restrict_types


class ImportVectorPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        input_folder_key: str = Field("input_data", description="The folder key into which the EOPatch will be saved. ")
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        input_filename: str = Field(
            description="Filename of the vector file to be imported. Needs to be located in the input-data folder."
        )
        reproject: bool = Field(True, description="Controls whether the vector file is reprojected to the EOPatch CRS.")
        clip: bool = Field(True, description="Controls whether the polygons are clipped to the EOPatch boundaries. ")
        output_feature: Feature = Field(description="The EOPatch feature to which the vector will be imported.")
        output_folder_key: str = Field(description="The folder key into which the EOPatch will be saved. ")
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        _restrict_output_feature = field_validator(
            "output_feature", restrict_types([FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS])
        )

    config: Schema

    def get_execution_arguments(self, workflow: EOWorkflow, patch_list: PatchList) -> ExecKwargs:
        exec_kwargs = super().get_execution_arguments(workflow, patch_list)
        vector_node = next(node for node in workflow.get_nodes() if isinstance(node.task, VectorImportTask))

        for name, bbox in patch_list:
            exec_kwargs[name][vector_node] = dict(bbox=bbox)

        return exec_kwargs

    def build_workflow(self) -> EOWorkflow:
        path = fs.path.join(self.storage.get_input_data_folder(), self.config.input_filename)

        vector_import_task = VectorImportTask(
            feature=self.config.output_feature,
            path=path,
            reproject=self.config.reproject,
            clip=self.config.clip,
            filesystem=self.storage.filesystem,
        )
        import_node = EONode(vector_import_task)

        save_task = SaveTask(
            path=self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            features=[self.config.output_feature],
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            use_zarr=self.storage.config.use_zarr,
        )
        save_node = EONode(save_task, inputs=[import_node])
        return EOWorkflow.from_endnodes(save_node)
