"""Implementation of the base Pipeline class."""

from __future__ import annotations

import ray
from pydantic import Field, ValidationError

from ..utils.meta import collect_schema, load_pipeline_class
from .config import RawConfig
from .schemas import BaseSchema


class PipelineRunSchema(BaseSchema):
    pipeline_config: dict
    ray_remote_kwargs: dict = Field(default_factory=dict)


def validate_chain(pipeline_chain: list[RawConfig]):
    for i, run_config in enumerate(pipeline_chain):
        try:
            run_schema = PipelineRunSchema.parse_obj(run_config)
        except ValidationError as e:
            raise TypeError(
                f"Pipeline-chain element {i} should be a dictionary with the fields `pipeline_config` and the optional"
                " `ray_remote_kwargs`."
            ) from e

        pipeline_schema = collect_schema(load_pipeline_class(run_schema.pipeline_config))
        pipeline_schema.parse_obj(run_schema.pipeline_config)


def run_pipeline_chain(pipeline_chain: list[RawConfig]):
    for run_config in pipeline_chain:
        run_schema = PipelineRunSchema.parse_obj(run_config)
        runner = _pipeline_runner.options(**run_schema.ray_remote_kwargs)
        ray.get(runner.remote(run_schema.pipeline_config))


@ray.remote
def _pipeline_runner(config: RawConfig):
    return load_pipeline_class(config).from_raw_config(config).run()
