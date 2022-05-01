"""
Pipelines for testing
"""
import logging
from typing import List, Tuple, Type, TypeVar, Optional, Dict, Any

import numpy as np
from pydantic import Field

from eolearn.core import EOWorkflow, CreateEOPatchTask, SaveTask, EONode, MergeEOPatchesTask, OverwritePermission

from ..core.config import RawConfig, recursive_config_join
from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..tasks.testing import DummyRasterFeatureTask, DummyTimestampFeatureTask
from ..utils.types import Feature, TimePeriod
from ..utils.validators import field_validator, parse_time_period

Self = TypeVar("Self", bound=Pipeline)
LOGGER = logging.getLogger(__name__)


class TestPipeline(Pipeline):
    """Pipeline that just tests if all managers works correctly. It can be used to check if area manager creates a
    correct grid.
    """

    class Schema(Pipeline.Schema):
        class Config:
            extra = "allow"

    _DEFAULT_CONFIG_PARAMS = {
        "pipeline": "eogrow.pipelines.testing.TestPipeline",
        "eopatch": {"manager": "eogrow.eopatches.EOPatchManager"},
        "logging": {"manager": "eogrow.logging.LoggingManager", "show_logs": True},
    }

    @classmethod
    def with_defaults(cls: Type[Self], config: RawConfig) -> Self:
        config = recursive_config_join(config, cls._DEFAULT_CONFIG_PARAMS)  # type: ignore
        return cls.from_raw_config(config)

    def run_procedure(self) -> Tuple[List, List]:
        """Performs basic tests of managers"""
        if self.storage.filesystem.exists("/"):
            LOGGER.info("Project folder %s exists", self.storage.config.project_folder)
        else:
            LOGGER.info("Project folder %s does not exist", self.storage.config.project_folder)

        self.area_manager.get_area_dataframe()
        self.area_manager.get_area_geometry()
        grid = self.area_manager.get_grid()
        grid_size = self.area_manager.get_grid_size()
        LOGGER.info("Grid has %d EOPatches and is split over %d CRS zones", grid_size, len(grid))

        eopatches = self.eopatch_manager.get_eopatch_filenames()
        LOGGER.info("The first EOPatch has a name %s", eopatches[0])

        return [], []


class RasterFeatureSchema(BaseSchema):
    feature: Feature = Field(description="A feature to be processed.")
    shape: Tuple[int, ...] = Field(description="A shape of a feature")
    dtype: Optional[str] = Field(description="The output dtype of the feature")
    min_value: int = Field(0, description="All values in the feature will be greater or equal to this value.")
    max_value: int = Field(1, description="All values in the feature will be smaller to this value.")


class TimestampFeatureSchema(BaseSchema):
    time_period: TimePeriod = Field(description="Time period from where timestamps will be generated.")
    _validate_time_period = field_validator("time_period", parse_time_period, pre=True)

    timestamp_num: int = Field(description="Number of timestamps from the interval")


class DummyDataPipeline(Pipeline):
    """Pipeline for generating dummy data."""

    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(description="The storage manager key pointing to the pipeline output folder.")
        seed: Optional[int] = Field(description="A randomness seed.")

        raster_features: List[RasterFeatureSchema]
        timestamp_feature: Optional[TimestampFeatureSchema]

    config: Schema

    def build_workflow(self) -> EOWorkflow:
        start_node = EONode(CreateEOPatchTask())

        if self.config.timestamp_feature:
            task = DummyTimestampFeatureTask(
                time_interval=self.config.timestamp_feature.time_period,
                timestamp_num=self.config.timestamp_feature.timestamp_num,
            )
            start_node = EONode(task, inputs=[start_node])

        add_feature_nodes = []
        for feature_config in self.config.raster_features:
            task = DummyRasterFeatureTask(
                feature_config.feature,
                shape=feature_config.shape,
                dtype=np.dtype(feature_config.dtype) if feature_config.dtype else None,
                min_value=feature_config.min_value,
                max_value=feature_config.max_value,
            )
            node = EONode(task, inputs=[start_node])
            add_feature_nodes.append(node)

        join_node = EONode(MergeEOPatchesTask(), inputs=add_feature_nodes)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key, full_path=True),
            overwrite_permission=OverwritePermission.OVERWRITE_PATCH,
            config=self.sh_config,
        )
        save_node = EONode(save_task, inputs=[join_node])

        return EOWorkflow.from_endnodes(save_node)

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, Any]]]:
        """Extends the basic method for adding execution arguments by adding seed arguments a sampling task"""
        exec_args = super().get_execution_arguments(workflow)

        add_feature_nodes = [
            node
            for node in workflow.get_nodes()
            if isinstance(node.task, (DummyRasterFeatureTask, DummyTimestampFeatureTask))
        ]
        generator = np.random.default_rng(seed=self.config.seed)

        for workflow_args in exec_args:
            for node in add_feature_nodes:
                workflow_args[node] = dict(seed=generator.integers(low=0, high=2**32))

        return exec_args
