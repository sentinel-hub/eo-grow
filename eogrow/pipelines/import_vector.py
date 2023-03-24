"""Implements a pipeline for importing vector data from a file."""


import fs
from pydantic import Field

from eolearn.core import EONode, EOPatch, EOTask, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.core.constants import TIMESTAMP_COLUMN
from eolearn.io import VectorImportTask

from ..core.pipeline import Pipeline
from ..types import ExecKwargs, Feature, PatchList
from ..utils.validators import field_validator, restrict_types


class ExtractTimestampsTask(EOTask):
    def __init__(self, input_feature: Feature):
        self.input_feature = input_feature

    def execute(self, eopatch: EOPatch) -> EOPatch:
        gdf = eopatch[self.input_feature]
        eopatch.timestamps = list(gdf[TIMESTAMP_COLUMN].unique())
        return eopatch


class ImportVectorPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        input_folder_key: str = Field("input_data", description="The folder key into which the EOPatch will be saved. ")
        input_filename: str = Field(
            description="Filename of the vector file to be imported. Needs to be located in the input-data folder."
        )
        reproject: bool = Field(True, description="Controls whether the vector file is reprojected to the EOPatch CRS.")
        clip: bool = Field(True, description="Controls whether the polygons are clipped to the EOPatch boundaries. ")
        output_feature: Feature = Field(description="The EOPatch feature to which the vector will be imported.")
        output_folder_key: str = Field(description="The folder key into which the EOPatch will be saved. ")

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
        previous_node = EONode(vector_import_task)

        if self.config.output_feature[0].is_temporal():
            previous_node = EONode(ExtractTimestampsTask(self.config.output_feature), inputs=[previous_node])

        save_task = SaveTask(
            path=self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            features=[FeatureType.BBOX, FeatureType.TIMESTAMPS, self.config.output_feature],
            compress_level=1,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
        )
        save_node = EONode(save_task, inputs=[previous_node])
        return EOWorkflow.from_endnodes(save_node)
