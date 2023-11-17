"""Implementation of the base Pipeline class."""

from __future__ import annotations

from ..utils.meta import collect_schema, load_pipeline_class
from .config import RawConfig


def validate_chain(pipeline_chain: list[RawConfig]):
    for config in pipeline_chain:
        pipeline_schema = collect_schema(load_pipeline_class(config))
        pipeline_schema.parse_obj(config)
