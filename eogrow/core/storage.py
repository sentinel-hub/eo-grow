"""Implementation of the StorageManager class for handling project storage."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Literal, Optional

import fs
from pydantic import BaseSettings, Field

import sentinelhub
from eolearn.core.utils.fs import get_aws_credentials, get_filesystem, is_s3_path
from sentinelhub import SHConfig

from .base import EOGrowObject
from .schemas import ManagerSchema


class StorageManager(EOGrowObject):
    PRESET_FOLDERS: ClassVar[dict[str, str]] = {"logs": "logs", "input_data": "input-data", "cache": "cache"}

    class Schema(ManagerSchema, BaseSettings):
        project_folder: str = Field(
            description=(
                "The root project folder. Can be either local or on AWS S3 Bucket."
                "If on AWS, the path must be prefixed with s3://."
            ),
        )
        aws_profile: Optional[str] = Field(
            env="AWS_PROFILE",
            description=(
                "The AWS profile with credentials needed to access the S3 buckets. In case the profile isn't specified"
                " with a parameter it can be read from an environmental variable."
            ),
        )
        filesystem_kwargs: Dict[str, Any] = Field(
            default_factory=dict, description="Optional kwargs to be passed on to FS specs."
        )
        structure: Dict[str, str] = Field(
            default_factory=dict,
            description="A flat key: value store mapping each key to a path in the project.",
        )
        geopandas_backend: Literal["fiona", "pyogrio"] = Field(
            "fiona", description="Which backend is used for IO operations when using geopandas."
        )
        use_zarr: bool = Field(False, description="Use the Zarr backend for EOPatch IO.")

        class Config(ManagerSchema.Config):
            case_sensitive = True
            env_prefix = "eogrow_"

    config: Schema

    def __init__(self, config: Schema):
        super().__init__(config)

        for folder_key, folder_path in self.PRESET_FOLDERS.items():
            if folder_key not in self.config.structure:
                self.config.structure[folder_key] = folder_path

        self.sh_config = self._prepare_sh_config()
        self.filesystem = get_filesystem(
            self.config.project_folder, create=True, config=self.sh_config, **self.config.filesystem_kwargs
        )

    def _prepare_sh_config(self) -> SHConfig:
        """Prepares an instance of `SHConfig` containing AWS credentials. In case given AWS profile doesn't exist it
        will show a warning and return a config without AWS credentials."""
        sh_config = SHConfig(hide_credentials=True) if sentinelhub.__version__ < "3.9.0" else SHConfig()

        if self.is_on_s3() and self.config.aws_profile:
            sh_config = get_aws_credentials(aws_profile=self.config.aws_profile, config=sh_config)

        return sh_config

    def get_folder(self, key: str, full_path: bool = False) -> str:
        """Returns the path associated with the given key in the structure config."""
        folder_path = self.config.structure[key]
        self.filesystem.makedirs(folder_path, recreate=True)

        if full_path:
            return fs.path.combine(self.config.project_folder, folder_path)
        return folder_path

    def get_logs_folder(self, full_path: bool = False) -> str:
        """Method for obtaining the logs folder."""
        return self.get_folder("logs", full_path=full_path)

    def get_cache_folder(self, full_path: bool = False) -> str:
        """Returns the path associated with the cache key."""
        return self.get_folder("cache", full_path=full_path)

    def get_input_data_folder(self, full_path: bool = False) -> str:
        """Returns the path associated with the input_data key."""
        return self.get_folder("input_data", full_path=full_path)

    def is_on_s3(self) -> bool:
        """Returns True if the project_folder is on S3, False otherwise."""
        return is_s3_path(self.config.project_folder)
