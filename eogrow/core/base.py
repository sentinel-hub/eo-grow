from typing import Type
from pydantic import BaseModel

from .config import Config
from ..utils.meta import collect_schema


class EOGrowObject:
    """A base object object in `eo-grow` framework"""

    def __init__(self, config: dict):
        config = config if isinstance(config, Config) else Config.from_dict(config)

        self.schema = self._initialize_schema()
        self._config = self._prepare_config(config)

    def _initialize_schema(self) -> Type[BaseModel]:
        schema_object = collect_schema(self)
        return schema_object

    def _prepare_config(self, config: Config) -> Config:
        """Interprets and validates configuration dictionary"""
        config = config.interpret()
        parsed_config = self.schema.parse_obj(config)

        return Config.from_dict(parsed_config.dict())

    @property
    def config(self) -> Config:
        """A public property that provides object's configuration"""
        return self._config
