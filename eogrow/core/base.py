"""Base object from which all configurable eo-grow objects inherit."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from .config import RawConfig, interpret_config_from_path

Self = TypeVar("Self", bound="EOGrowObject")


class EOGrowObject:
    """A base object in `eo-grow` framework"""

    class Schema(BaseModel):
        """A pydantic parsing/validation schema describing the shape of input parameters."""

        class Config:
            """Forbids unspecified fields and validates default values as well."""

            extra = "forbid"
            validate_all = True
            allow_mutation = False
            arbitrary_types_allowed = True

    config: Schema

    def __init__(self, config: Schema):
        self.config = config

    @classmethod
    def from_raw_config(cls: type[Self], config: RawConfig, *args: Any, **kwargs: Any) -> Self:
        """Creates an object from a dictionary by constructing a validated config and use it to create the object."""
        validated_config = cls.Schema.parse_obj(config)
        return cls(validated_config, *args, **kwargs)

    @classmethod
    def from_path(cls: type[Self], path: str, *args: Any, **kwargs: Any) -> Self:
        """Creates an object by loading and validating a config from a JSON file."""
        config = interpret_config_from_path(path)
        return cls.from_raw_config(config, *args, **kwargs)
