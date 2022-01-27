"""
Pipelines that transform data
"""
from typing import Dict
from pydantic import Field
from eolearn.core import LoadTask, EOWorkflow, linearly_connect_tasks, SaveTask, FeatureType, OverwritePermission

from ..core.pipeline import Pipeline
from ..tasks.common import MappingTask


class MappingPipeline(Pipeline):
    """A pipeline for transforming a values of a mask_timeless feature according to given mapping rules"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(description="The storage manager key pointing to the input folder")
        input_feature: str = Field(description="Name of the input mask_timeless feature")
        output_folder_key: str = Field(description="The storage manager key pointing to the output folder")
        output_feature: str = Field(description="Name of the output mask_timeless feature")

        mapping_dictionary: Dict[int, int] = Field(description="Mapping dictionary of the input-to-output classes")
        compress_level: int = Field(1, description="Level of compression used in saving eopatches")

    def build_workflow(self) -> EOWorkflow:
        """Method for constructing the workflow"""
        input_feature = FeatureType.MASK_TIMELESS, self.config.input_feature
        output_feature = FeatureType.MASK_TIMELESS, self.config.output_feature
        extra_features = [FeatureType.BBOX] if self.config.input_folder_key != self.config.output_folder_key else []

        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key, full_path=True),
            features=[input_feature] + extra_features,
            config=self.sh_config,
        )

        mapping_task = MappingTask(input_feature, output_feature, self.config.mapping_dictionary)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key, full_path=True),
            features=[output_feature] + extra_features,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
            compress_level=self.config.compress_level,
        )

        return EOWorkflow(linearly_connect_tasks(load_task, mapping_task, save_task))
