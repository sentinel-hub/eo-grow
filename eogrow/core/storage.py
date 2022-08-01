"""
This module handles everything regarding storage of the data
"""
from io import StringIO
from typing import Dict, List, Optional

import fs
from fs.base import FS
from pydantic import BaseSettings, Field

from eolearn.core.utils.fs import get_aws_credentials, get_filesystem, is_s3_path
from sentinelhub import SHConfig

from ..utils.types import AwsAclType
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
            default_factory=dict, description="A flat key: value store mapping each key to a path in the project."
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
        """Prepares the main instance of filesystem object which contains all additional configuration parameters."""
        fs_kwargs: Dict[str, str] = {}
        if is_s3_path(self.config.project_folder) and self.config.aws_acl:
            fs_kwargs["acl"] = self.config.aws_acl

        return get_filesystem(self.config.project_folder, create=True, config=self.sh_config, **fs_kwargs)

    def get_folder(self, key: str, full_path: bool = False) -> str:
        """Returns the path  associated with a key in the structure config."""
        folder_path = self.config.structure[key]
        self.filesystem.makedirs(folder_path, recreate=True)

        if full_path:
            return fs.path.combine(self.config.project_folder, folder_path)
        return folder_path

    def get_logs_folder(self, full_path: bool = False) -> str:
        """Method for obtaining the logs folder. Will store logs to the current folder.
        Temporary solution until the logging to AWS is handled properly
        """
        return self.get_folder("logs", full_path=full_path)

    def get_cache_folder(self, full_path: bool = False) -> str:
        """Returns the path associated with the cache key."""
        return self.get_folder("cache", full_path=full_path)

    def get_input_data_folder(self, full_path: bool = False) -> str:
        """Returns the path associated with the input_data key."""
        return self.get_folder("input_data", full_path=full_path)

    def is_on_aws(self) -> bool:
        """Returns True if the project_folder is on S3, False  otherwise."""
        return is_s3_path(self.config.project_folder)

    def show_folder_structure(
        self, show_files: bool = False, return_str: bool = False, exclude: Optional[List[str]] = None
    ) -> Optional[str]:
        """Shows how folder structure looks like at the moment. It will show all folders except EOPatch folders and
        EOExecution report folders

        :param show_files: If  `True` it will show also files inside the folders. Note that the number of files may be
            huge. By default, this is set to `False`.
        :param return_str: If `True` it will return folder structure as a string. If `False` it will just print the
            visualization to stdout.
        :param exclude: A list of grep folder paths  to exclude from the structure return.
            Defaults to ['eopatch*', 'eoexecution-report*']
        :return: Depending on return_str it will either return a string or None
        """
        if exclude is None:
            exclude = ["eopatch*", "eoexecution-report*"]

        file_filter = None if show_files else [""]
        io_object = StringIO() if return_str else None
        self.filesystem.tree(max_levels=10, with_color=True, exclude=exclude, filter=file_filter, file=io_object)
        if io_object:
            io_object.seek(0)
            return io_object.getvalue()
        return None
