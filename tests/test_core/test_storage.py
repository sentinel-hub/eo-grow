""" A module for testing StorageManager class
"""
import os

import pytest
from fs.osfs import OSFS

from eogrow.core.config import Config
from eogrow.core.storage import StorageManager

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="local_storage_manager")
def local_storage_manager_fixture(config_folder):
    filename = os.path.join(config_folder, "other", "local_storage_test.json")
    config = Config.from_path(filename)
    return StorageManager(config.storage)


@pytest.fixture(scope="session", name="aws_storage_manager")
def aws_storage_manager(project_folder, config_folder):
    filename = os.path.join(config_folder, "other", "aws_storage_test.json")
    config = Config.from_path(filename)
    config.storage.project_folder = project_folder
    return StorageManager(config.storage)


@pytest.fixture(scope="session", name="aws_storage_config_fixture")
def aws_storage_config_fixture(config_folder):
    filename = os.path.join(config_folder, "other", "aws_storage_test.json")
    return Config.from_path(filename)


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
