"""
Module with global fixtures
"""
import os
import shutil
from tempfile import TemporaryDirectory

import pytest

from eogrow.core.config import interpret_config_from_path
from eogrow.core.storage import StorageManager


@pytest.fixture(scope="session", name="project_folder")
def project_folder_fixture():
    """Folder of the test project"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_project")


@pytest.fixture(scope="session", name="config_folder")
def config_folder_fixture():
    """Folder with configs"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_config_files")


@pytest.fixture(scope="session", name="stats_folder")
def stats_folder_fixture():
    """Folder with stats"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_stats")


@pytest.fixture(name="temp_folder")
def temp_folder_fixture():
    """A temporary folder"""
    with TemporaryDirectory() as temp_folder:
        yield temp_folder


@pytest.fixture(scope="session", name="test_storage_manager")
def test_storage_manager_fixture(project_folder):
    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_config_files", "other", "local_storage_test.json"
    )
    config = interpret_config_from_path(filename)
    config["storage"]["project_folder"] = project_folder
    yield StorageManager.from_raw_config(config["storage"])


@pytest.fixture(name="storage")
def storage_fixture(test_storage_manager):
    _clear_test_project_folder(test_storage_manager)

    yield test_storage_manager

    # This is a teardown stage:
    _clear_test_project_folder(test_storage_manager)


def _clear_test_project_folder(storage):
    for content in os.listdir(storage.config.project_folder):
        if content != "input-data":
            shutil.rmtree(os.path.join(storage.config.project_folder, content))


@pytest.fixture(scope="session", name="config")
def config_fixture(config_folder):
    path = os.path.join(config_folder, "global_config.json")
    return interpret_config_from_path(path)
