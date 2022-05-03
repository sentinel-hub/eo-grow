"""
Pipelines for testing
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
from pydantic import Field

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, MergeEOPatchesTask, OverwritePermission, SaveTask

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


class FeatureSchema(BaseSchema):
    same_for_all: bool = Field(
        False,
        description=(
            "A flag to specify if the same feature values should be generated for all EOPatches. By default each"
            " EOPatch will have different values."
        ),
    )


class RasterFeatureSchema(FeatureSchema):
    feature: Feature = Field(description="A feature to be processed.")
    shape: Tuple[int, ...] = Field(description="A shape of a feature")
    dtype: str = Field(description="The output dtype of the feature")
    min_value: int = Field(0, description="All values in the feature will be greater or equal to this value.")
    max_value: int = Field(1, description="All values in the feature will be smaller to this value.")


class TimestampFeatureSchema(FeatureSchema):
    time_period: TimePeriod = Field(description="Time period from where timestamps will be generated.")
    _validate_time_period = field_validator("time_period", parse_time_period, pre=True)

    timestamp_num: int = Field(description="Number of timestamps from the interval")


class DummyDataPipeline(Pipeline):
    """Pipeline for generating dummy data."""

    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(description="The storage manager key pointing to the pipeline output folder.")
        seed: Optional[int] = Field(description="A randomness seed.")

        raster_features: List[RasterFeatureSchema] = Field(
            default_factory=list, description="A list of raster features to be generated."
        )
        timestamp_feature: Optional[TimestampFeatureSchema]

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._nodes_to_configs_map: Dict[EONode, FeatureSchema] = {}

    def build_workflow(self) -> EOWorkflow:
        """Creates a workflow with tasks that generate different types of features and tasks that join and save the
        final EOPatch."""
        self._nodes_to_configs_map = {}
        start_node = EONode(CreateEOPatchTask())

        if self.config.timestamp_feature:
            task = DummyTimestampFeatureTask(
                time_interval=self.config.timestamp_feature.time_period,
                timestamp_num=self.config.timestamp_feature.timestamp_num,
            )
            start_node = EONode(task, inputs=[start_node])
            self._nodes_to_configs_map[start_node] = self.config.timestamp_feature

        add_feature_nodes = []
        for index, feature_config in enumerate(self.config.raster_features):
            task = DummyRasterFeatureTask(
                feature_config.feature,
                shape=feature_config.shape,
                dtype=np.dtype(feature_config.dtype),
                min_value=feature_config.min_value,
                max_value=feature_config.max_value,
            )
            node = EONode(task, inputs=[start_node], name=f"{DummyRasterFeatureTask.__name__}_{index}")
            add_feature_nodes.append(node)
            self._nodes_to_configs_map[node] = feature_config

        if add_feature_nodes:
            join_node = EONode(MergeEOPatchesTask(), inputs=add_feature_nodes)
            previous_node = join_node
        else:
            previous_node = start_node

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key, full_path=True),
            overwrite_permission=OverwritePermission.OVERWRITE_PATCH,
            config=self.sh_config,
        )
        save_node = EONode(save_task, inputs=[previous_node])

        return EOWorkflow.from_endnodes(save_node)

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        """Extends the basic method for adding execution arguments by adding seed arguments a sampling task"""
        exec_args = super().get_execution_arguments(workflow)

        # Sorting is done to ensure seeds are always given to nodes in the same order
        add_feature_nodes = sorted(self._nodes_to_configs_map, key=lambda _node: _node.get_name())

        generator = np.random.default_rng(seed=self.config.seed)
        for index, workflow_args in enumerate(exec_args):
            for node in add_feature_nodes:
                seed = generator.integers(low=0, high=2**32)

                if self._nodes_to_configs_map[node].same_for_all and index > 0:
                    seed = exec_args[0][node]["seed"]  # type: ignore

                workflow_args[node] = dict(seed=seed)

        return exec_args
