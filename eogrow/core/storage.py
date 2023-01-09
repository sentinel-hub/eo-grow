"""Implementation of the StorageManager class for handling project storage."""
from typing import Dict, Literal, Optional

import fs
from fs.base import FS
from pydantic import BaseSettings, Field

from eolearn.core.utils.fs import get_aws_credentials, get_filesystem, is_s3_path
from sentinelhub import SHConfig

from ..types import AwsAclType
from .base import EOGrowObject
from .schemas import ManagerSchema


class StorageManager(EOGrowObject):
    PRESET_FOLDERS: Dict[str, str] = {"logs": "logs", "input_data": "input-data", "cache": "cache"}

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
        aws_acl: Optional[AwsAclType] = Field(
            description=(
                "An optional parameter to specify under what kind of access control list (ACL) objects should be saved"
                " to an AWS S3 bucket."
            )
        )
        structure: Dict[str, str] = Field(
            default_factory=dict,
            description="A flat key: value store mapping each key to a path in the project.",
        )
        geopandas_backend: Literal["fiona", "pyogrio"] = Field(
            "fiona", description="Which backend is used for IO operations when using geopandas."
        )

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
        self.filesystem = self._prepare_filesystem()

    def _prepare_sh_config(self) -> SHConfig:
        """Prepares an instance of `SHConfig` containing AWS credentials. In case given AWS profile doesn't exist it
        will show a warning and return a config without AWS credentials."""
        sh_config = SHConfig(hide_credentials=True)

        if self.is_on_aws() and self.config.aws_profile:
            sh_config = get_aws_credentials(aws_profile=self.config.aws_profile, config=sh_config)

        return sh_config

    def _prepare_filesystem(self) -> FS:
        """Prepares a filesystem object with the configuration parameters."""
        fs_kwargs: Dict[str, str] = {}
        if is_s3_path(self.config.project_folder) and self.config.aws_acl:
            fs_kwargs["acl"] = self.config.aws_acl

        return get_filesystem(self.config.project_folder, create=True, config=self.sh_config, **fs_kwargs)

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

    def is_on_aws(self) -> bool:
        """Returns True if the project_folder is on S3, False otherwise."""
        return is_s3_path(self.config.project_folder)
