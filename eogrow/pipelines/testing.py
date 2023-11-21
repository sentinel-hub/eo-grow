"""Implements pipelines used for data preparation in testing."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import Field

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, OverwritePermission, SaveTask
from eolearn.core.types import Feature

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..tasks.testing import GenerateRasterFeatureTask, GenerateTimestampsTask, NormalDistribution, UniformDistribution
from ..types import ExecKwargs, PatchList, TimePeriod
from ..utils.validators import ensure_storage_key_presence, field_validator, parse_dtype, parse_time_period


class UniformDistributionSchema(BaseSchema):
    kind: Literal["uniform"]
    min_value: float = Field(0, description="All values in the feature will be greater or equal to this value.")
    max_value: float = Field(1, description="All values in the feature will be smaller or equal than this value.")


class NormalDistributionSchema(BaseSchema):
    kind: Literal["normal"]
    mean: float = Field(0, description="Mean of the normal distribution.")
    std: float = Field(1, description="Standard deviation of the normal distribution.")


class RasterFeatureGenerationSchema(BaseSchema):
    feature: Feature = Field(description="Feature to be created.")
    shape: Tuple[int, ...] = Field(description="Shape of the feature")
    dtype: np.dtype = Field(description="The output dtype of the feature")
    _parse_dtype = field_validator("dtype", parse_dtype, pre=True)
    distribution: Union[UniformDistributionSchema, NormalDistributionSchema] = Field(
        description="Choice of distribution for generating values.", discriminator="kind"
    )


class TimestampGenerationSchema(BaseSchema):
    time_period: TimePeriod = Field(description="Time period from where timestamps will be generated.")
    _validate_time_period = field_validator("time_period", parse_time_period, pre=True)

    num_timestamps: int = Field(description="Number of timestamps from the interval")
    same_for_all: bool = Field(True, description="Whether all EOPatches should have the same timestamps")


class GenerateDataPipeline(Pipeline):
    """Pipeline for generating test input data."""

    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(description="The storage manager key pointing to the pipeline output folder.")
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        seed: int = Field(description="A seed with which per-eopatch RNGs seeds are generated.")

        features: List[RasterFeatureGenerationSchema] = Field(
            default_factory=list, description="A specification for features to be generated."
        )
        timestamps: Optional[TimestampGenerationSchema]
        meta_info: Optional[dict] = Field(
            description="Information to be stored into the meta-info fields of each EOPatch."
        )

    config: Schema

    def build_workflow(self) -> EOWorkflow:
        """Creates a workflow with tasks that generate different types of features and tasks that join and save the
        final EOPatch."""
        previous_node = EONode(CreateEOPatchTask())

        if self.config.timestamps:
            timestamp_task = GenerateTimestampsTask(
                time_interval=self.config.timestamps.time_period, num_timestamps=self.config.timestamps.num_timestamps
            )
            previous_node = EONode(timestamp_task, inputs=[previous_node])

        for feature_config in self.config.features:
            raster_task = GenerateRasterFeatureTask(
                feature_config.feature,
                shape=feature_config.shape,
                dtype=np.dtype(feature_config.dtype),
                distribution=self._convert_distribution_configuration(feature_config.distribution),
            )
            previous_node = EONode(raster_task, inputs=[previous_node], name=str(feature_config.feature))

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            save_timestamps=self.config.timestamps is not None,
            use_zarr=self.storage.config.use_zarr,
        )
        save_node = EONode(save_task, inputs=[previous_node])

        return EOWorkflow.from_endnodes(save_node)

    def _convert_distribution_configuration(
        self, distribution_config: NormalDistributionSchema | UniformDistributionSchema
    ) -> NormalDistribution | UniformDistribution:
        if isinstance(distribution_config, NormalDistributionSchema):
            return NormalDistribution(distribution_config.mean, distribution_config.std)
        return UniformDistribution(distribution_config.min_value, distribution_config.max_value)

    def get_execution_arguments(self, workflow: EOWorkflow, patch_list: PatchList) -> ExecKwargs:
        """Extends the basic method for adding execution arguments by adding seed arguments a sampling task"""
        exec_args = super().get_execution_arguments(workflow, patch_list)

        rng = np.random.default_rng(seed=self.config.seed)
        per_node_seeds = {node: rng.integers(low=0, high=2**32) for node in workflow.get_nodes()}
        same_timestamps = self.config.timestamps and self.config.timestamps.same_for_all

        for node, node_seed in per_node_seeds.items():
            if isinstance(node.task, CreateEOPatchTask):
                for patch_args in exec_args.values():
                    patch_args[node]["meta_info"] = self.config.meta_info
            if isinstance(node.task, GenerateTimestampsTask) and same_timestamps:
                for patch_args in exec_args.values():
                    patch_args[node] = dict(seed=node_seed)
            elif isinstance(node.task, (GenerateRasterFeatureTask, GenerateTimestampsTask)):
                node_rng = np.random.default_rng(seed=node_seed)
                for patch_args in exec_args.values():
                    patch_args[node] = dict(seed=node_rng.integers(low=0, high=2**32))

        return exec_args
