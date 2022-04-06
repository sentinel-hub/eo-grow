"""Base object from which all configurable eo-grow objects inherit."""
from typing import Optional, Type, TypeVar

from .config import Config, prepare_config
from .schemas import BaseSchema

Self = TypeVar("Self", bound="EOGrowObject")


class EOGrowObject:
    """A base object object in `eo-grow` framework"""

    class Schema(BaseSchema):
        """A pydantic parsing/validation schema describing the shape of input parameters."""

    def __init__(self, config: Config, unvalidated_config: Optional[Config] = None):
        self._config = config
        self._unvalidated_config = unvalidated_config

    @classmethod
    def from_raw_config(cls: Type[Self], config: Config) -> Self:
        validated_config = prepare_config(config, cls.Schema)
        return cls(validated_config, config)

    @property
    def config(self) -> Config:
        """A public property that provides object's configuration"""
        return self._config
