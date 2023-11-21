"""Module implementing utilities for chained configs."""

from __future__ import annotations

from typing import Any, Dict

import ray
from pydantic import Field, ValidationError

from ..core.config import RawConfig
from ..core.schemas import BaseSchema
from .meta import collect_schema, load_pipeline_class


class PipelineRunSchema(BaseSchema):
    pipeline_config: dict
    pipeline_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to ray when executing the main pipeline process. The options are specified [here]"
            "(https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html)."
        ),
    )


def validate_pipeline_chain(pipeline_chain: list[RawConfig]) -> None:
    for i, run_config in enumerate(pipeline_chain):
        try:
            run_schema = PipelineRunSchema.parse_obj(run_config)
        except ValidationError as e:
            raise TypeError(
                f"Pipeline-chain element {i} should be a dictionary with the fields `pipeline_config` and the optional"
                " `pipeline_resources`."
            ) from e

        pipeline_schema = collect_schema(load_pipeline_class(run_schema.pipeline_config))
        pipeline_schema.parse_obj(run_schema.pipeline_config)


def run_pipeline_chain(pipeline_chain: list[RawConfig]) -> None:
    for run_config in pipeline_chain:
        run_schema = PipelineRunSchema.parse_obj(run_config)
        runner = _pipeline_runner.options(**run_schema.pipeline_resources)  # type: ignore[attr-defined]
        ray.get(runner.remote(run_schema.pipeline_config))


@ray.remote
def _pipeline_runner(config: RawConfig) -> None:
    return load_pipeline_class(config).from_raw_config(config).run()
