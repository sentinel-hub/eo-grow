"""Utilities for working with lists of EOPatch names."""

from typing import List

import rapidjson
from fs.base import FS


def save_eopatch_names(filesystem: FS, file_path: str, eopatch_list: List[str]) -> None:
    """Saves a list of EOPatches to a file

    :param filesystem: Filesystem used to save the file.
    :param filename: Path of a JSON file where names of EOPatches will be saved.
    :param eopatch_list: A list of EOPatch names.
    """

    with filesystem.openbin(file_path, "w") as file:
        rapidjson.dump(eopatch_list, file, indent=2)


def load_eopatch_names(filesystem: FS, file_path: str) -> List[str]:
    """Loads a list of EOPatch names from a file

    :param filesystem: Filesystem used to load the file.
    :param filename: Path of a JSON file where names of EOPatches are saved.
    :return: A list of EOPatch names loaded from file.
    """
    with filesystem.openbin(file_path, "w") as file:
        return rapidjson.load(file)
