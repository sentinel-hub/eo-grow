"""
Tests for fs_utils module
"""
import json
import os

import pytest
from fs.tempfs import TempFS
from fs.walk import Walker

from eogrow.utils.fs import LocalFile, LocalFolder

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("always_copy", [True, False])
def test_local_file(always_copy):
    """Testing the procedure of saving and loading with LocalFile."""
    with TempFS() as filesystem:
        with LocalFile("path/to/file/data.json", mode="w", filesystem=filesystem, always_copy=always_copy) as test_file:
            with open(test_file.path, "w") as fp:
                json.dump({}, fp)

        with LocalFile("path/to/file/data.json", mode="r", filesystem=filesystem, always_copy=always_copy) as test_file:
            with open(test_file.path, "r") as fp:
                result = json.load(fp)

        assert result == {}


@pytest.mark.parametrize("always_copy", [True, False])
def test_copies_between_local_and_remote(always_copy):
    """Tests that `copy_to_remote` and `copy_to_local` work correctly."""
    with TempFS() as filesystem:
        remote_path = "folder/data.json"
        with LocalFile(remote_path, mode="w", filesystem=filesystem, always_copy=always_copy) as test_file:
            if always_copy:
                assert not filesystem.exists("folder")
            else:
                assert filesystem.exists("folder")

            with open(test_file.path, "w") as fp:
                json.dump({}, fp)
            if not always_copy:
                assert filesystem.isfile(remote_path)
            test_file.copy_to_remote()
            assert filesystem.isfile(remote_path)

            new_content = {"test": 1}
            with filesystem.open(remote_path, "w") as fp:
                json.dump(new_content, fp)
            test_file.copy_to_local()
            with open(test_file.path, "r") as fp:
                result = json.load(fp)
            assert result == new_content


@pytest.mark.parametrize("always_copy", [True, False])
@pytest.mark.parametrize("workers", [None, 0, 4])
@pytest.mark.parametrize("walker", [None, Walker(max_depth=1)])
def test_local_folder(always_copy, workers, walker):
    """Testing the procedure of saving and loading items with LocalFolder."""

    filenames = ["data1.json", os.path.join("subfolder", "data2.json")]

    with TempFS() as filesystem:
        with LocalFolder(
            "path/to/folder/", mode="w", filesystem=filesystem, always_copy=always_copy, workers=workers, walker=walker
        ) as test_folder:
            os.mkdir(os.path.join(test_folder.path, "subfolder"))

            for index, filename in enumerate(filenames):
                file_path = os.path.join(test_folder.path, filename)
                with open(file_path, "w") as fp:
                    json.dump({"value": index}, fp)

        with LocalFolder(
            "path/to/folder/", mode="r", filesystem=filesystem, always_copy=always_copy, workers=workers, walker=walker
        ) as test_folder:
            for index, filename in enumerate(filenames):
                if walker and walker.max_depth <= 1 and filename.startswith("subfolder"):
                    continue

                file_path = os.path.join(test_folder.path, filename)
                with open(file_path, "r") as fp:
                    result = json.load(fp)
                    assert result == {"value": index}
