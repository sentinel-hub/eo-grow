""" A module for testing StorageManager class
"""
import os
from typing import Optional

import pytest
from botocore.exceptions import ProfileNotFound
from fs.osfs import OSFS
from fs_s3fs import S3FS

from eogrow.core.config import RawConfig, interpret_config_from_path
from eogrow.core.storage import StorageManager

pytestmark = pytest.mark.fast


@pytest.fixture(name="local_storage_manager")
def local_storage_manager_fixture(config_folder):
    filename = os.path.join(config_folder, "other", "local_storage_test.json")
    config = interpret_config_from_path(filename)
    return StorageManager.from_raw_config(config["storage"])


@pytest.fixture(name="aws_storage_config")
def aws_pipeline_config_fixture(config_folder):
    filename = os.path.join(config_folder, "other", "aws_storage_test.json")
    return interpret_config_from_path(filename)


@pytest.fixture(name="aws_storage_manager")
def aws_storage_manager_fixture(aws_storage_config, project_folder):
    aws_storage_config["project_folder"] = project_folder
    return StorageManager.from_raw_config(aws_storage_config)


def test_storage_basic_local(local_storage_manager, project_folder):
    assert isinstance(local_storage_manager.filesystem, OSFS)
    assert local_storage_manager.filesystem.root_path == project_folder

    # TODO: Mocking AWS S3 and testing functionality if path is set to AWS.


def test_get_input_data_folder(local_storage_manager: StorageManager, project_folder):
    assert local_storage_manager.get_input_data_folder(full_path=False) == "input-data"

    input_data_path = os.path.join(project_folder, "input-data")
    assert os.path.normpath(local_storage_manager.get_input_data_folder(full_path=True)) == input_data_path


def test_get_cache_folder(local_storage_manager: StorageManager, project_folder):
    assert local_storage_manager.get_cache_folder(full_path=False) == "cache"

    cache_path = os.path.join(project_folder, "cache")
    assert local_storage_manager.get_cache_folder(full_path=True) == cache_path


def test_get_logs_folder(local_storage_manager: StorageManager, project_folder):
    assert local_storage_manager.get_logs_folder(full_path=False) == "logs"

    logs_path = os.path.join(project_folder, "logs")
    assert local_storage_manager.get_logs_folder(full_path=True) == logs_path


def test_get_custom_folder(local_storage_manager: StorageManager, project_folder):
    assert local_storage_manager.get_folder("eopatches", full_path=False) == "path/to/eopatches"

    abs_path = os.path.join(project_folder, "path", "to", "eopatches")
    assert local_storage_manager.get_folder("eopatches", full_path=True) == abs_path


@pytest.mark.parametrize("config_profile", [None, "", "nonexistent-config-profile"])
@pytest.mark.parametrize("env_profile", [None, "", "nonexistent-env-profile"])
def test_aws_profile(aws_storage_config: RawConfig, config_profile: Optional[str], env_profile: Optional[str]):
    """Checks different combinations of profile being set with a config parameter and environmental variable. Checks
    also that config parameter takes priority over environmental variable."""

    for parameter_key, parameter_value, config_dict in [
        ("aws_profile", config_profile, aws_storage_config),
        ("AWS_PROFILE", env_profile, os.environ),
    ]:
        if parameter_value is not None:
            config_dict[parameter_key] = parameter_value
        elif parameter_key in config_dict:
            del config_dict[parameter_key]

    expected_profile = config_profile if config_profile is not None else env_profile
    if expected_profile:
        with pytest.raises(ProfileNotFound) as exception_info:
            StorageManager.from_raw_config(aws_storage_config)

        assert str(exception_info.value) == f"The config profile ({expected_profile}) could not be found"
    else:
        storage = StorageManager.from_raw_config(aws_storage_config)
        assert storage.config.aws_profile == expected_profile


@pytest.mark.parametrize(
    "config",
    [
        {"project_folder": "s3://fake-bucket/", "aws_acl": "bucket-owner-full-control"},
        {"project_folder": "s3://fake-bucket/"},
        {"project_folder": ".", "aws_acl": "public-read"},
    ],
)
def test_aws_acl(config: RawConfig):
    storage = StorageManager.from_raw_config(config)

    if isinstance(storage.filesystem, S3FS):
        config_acl = config.get("aws_acl")
        filesystem_acl = None if storage.filesystem.upload_args is None else storage.filesystem.upload_args.get("ACL")
        assert config_acl == filesystem_acl
