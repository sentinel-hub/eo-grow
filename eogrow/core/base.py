"""Base object from which all configurable eo-grow objects inherit."""

from .config import Config


class EOGrowObject:
    """A base object object in `eo-grow` framework"""

    def __init__(self, config: Config):
        self._config = config

    @property
    def config(self) -> Config:
        """A public property that provides object's configuration"""
        return self._config
