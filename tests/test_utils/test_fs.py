"""
Tests for fs_utils module
"""
import json

import pytest
from fs.tempfs import TempFS

from eogrow.utils.fs import LocalFile

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("always_copy", [True, False])
def test_local_file(always_copy):
    with TempFS() as filesystem:
        with LocalFile("path/to/file/data.json", mode="w", filesystem=filesystem, always_copy=always_copy) as test_file:
            with open(test_file.path, "w") as fp:
                json.dump({}, fp)

        with LocalFile("path/to/file/data.json", mode="r", filesystem=filesystem, always_copy=always_copy) as test_file:
            with open(test_file.path, "r") as fp:
                result = json.load(fp)

        assert result == {}
