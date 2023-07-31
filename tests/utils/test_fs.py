import json
import os

import fs.path
import pytest
from fs.errors import ResourceNotFound
from fs.tempfs import TempFS
from fs.walk import Walker

from eogrow.utils.fs import LocalFile, LocalFolder


@pytest.mark.parametrize("always_copy", [True, False])
def test_local_file(always_copy):
    """Testing the procedure of saving and loading with LocalFile."""
    with TempFS() as filesystem:
        with LocalFile("path/to/file/data.json", mode="w", filesystem=filesystem, always_copy=always_copy) as test_file:
            assert not os.path.exists(test_file.path)

            with open(test_file.path, "w") as fp:
                json.dump({}, fp)

        tmp_file_still_exists = os.path.exists(test_file.path)
        assert tmp_file_still_exists is not always_copy

        with LocalFile("path/to/file/data.json", mode="r", filesystem=filesystem, always_copy=always_copy) as test_file:
            with open(test_file.path) as fp:
                result = json.load(fp)

        assert result == {}

        tmp_file_still_exists = os.path.exists(test_file.path)
        assert tmp_file_still_exists is not always_copy


def test_write_no_data_local_file():
    """Tests a case where no data is written to a local file. An error should only be raised if local file would be
    explicitly copied to remote."""
    with TempFS() as filesystem:
        remote_path = "folder/data.json"
        with LocalFile(remote_path, mode="w", filesystem=filesystem, always_copy=True) as test_file:
            test_file.copy_to_remote(raise_missing=False)
            with pytest.raises(ResourceNotFound):
                test_file.copy_to_remote(raise_missing=True)
        # A closure of "with statement" shouldn't raise an error or copy anything
        assert not filesystem.exists(remote_path)
        assert not filesystem.exists(fs.path.dirname(remote_path))


@pytest.mark.parametrize("always_copy", [True, False])
def test_copies_between_local_and_remote(always_copy):
    """Tests that `copy_to_remote` and `copy_to_local` work correctly."""
    with TempFS() as filesystem:
        remote_path = "folder/data.json"
        with LocalFile(remote_path, mode="w", filesystem=filesystem, always_copy=always_copy) as test_file:
            is_subfolder_created = filesystem.exists("folder")
            assert is_subfolder_created is not always_copy

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
            with open(test_file.path) as fp:
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
            assert os.path.exists(test_folder.path)
            os.mkdir(os.path.join(test_folder.path, "subfolder"))

            for index, filename in enumerate(filenames):
                file_path = os.path.join(test_folder.path, filename)
                with open(file_path, "w") as fp:
                    json.dump({"value": index}, fp)

        tmp_folder_still_exists = os.path.exists(test_folder.path)
        assert tmp_folder_still_exists is not always_copy

        with LocalFolder(
            "path/to/folder/", mode="r", filesystem=filesystem, always_copy=always_copy, workers=workers, walker=walker
        ) as test_folder:
            for index, filename in enumerate(filenames):
                if walker and walker.max_depth <= 1 and filename.startswith("subfolder"):
                    continue

                file_path = os.path.join(test_folder.path, filename)
                with open(file_path) as fp:
                    result = json.load(fp)
                    assert result == {"value": index}

        tmp_folder_still_exists = os.path.exists(test_folder.path)
        assert tmp_folder_still_exists is not always_copy


@pytest.mark.parametrize("use_absolute_path", [True, False])
def test_write_no_data_local_folder(use_absolute_path):
    """Tests a case where no data is written to a local folder."""
    relative_remote_path = "folder/data-folder"
    with TempFS() as filesystem:
        if use_absolute_path:
            params = dict(path=filesystem.getsyspath(relative_remote_path))
        else:
            params = dict(path=relative_remote_path, filesystem=filesystem)

        with LocalFolder(**params, mode="w", always_copy=True) as test_folder:
            test_folder.copy_to_remote()

        assert filesystem.isdir(relative_remote_path)


def test_path_identifier():
    """Checks that a temporary folder gets correct identifier suffix."""
    with TempFS() as filesystem:
        with LocalFile("data.json", mode="w", filesystem=filesystem, always_copy=True) as test_file:
            assert os.path.dirname(test_file.path).endswith("_LocalFile-data")

        with LocalFile("data.json", mode="w", filesystem=filesystem, always_copy=True, identifier="xxx") as test_file:
            assert os.path.dirname(test_file.path).endswith("xxx")

        with LocalFolder("my-folder", mode="w", filesystem=filesystem, always_copy=True) as test_folder:
            assert os.path.dirname(test_folder.path).endswith("_LocalFolder-my-folder")
