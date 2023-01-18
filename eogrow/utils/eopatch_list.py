"""Utilities for working with lists of EOPatch names."""

import json
from collections import defaultdict
from typing import DefaultDict, Dict, List

from fs.base import FS

from sentinelhub import CRS

from ..types import PatchList


def save_eopatch_names(filesystem: FS, file_path: str, eopatch_list: List[str]) -> None:
    """Saves a list of EOPatches to a file

    :param filesystem: Filesystem used to save the file.
    :param filename: Path of a JSON file where names of EOPatches will be saved.
    :param eopatch_list: A list of EOPatch names.
    """

    with filesystem.open(file_path, "w") as file:
        json.dump(eopatch_list, file, indent=2)


def load_eopatch_names(filesystem: FS, file_path: str) -> List[str]:
    """Loads a list of EOPatch names from a file

    :param filesystem: Filesystem used to load the file.
    :param filename: Path of a JSON file where names of EOPatches are saved.
    :return: A list of EOPatch names loaded from file.
    """
    with filesystem.open(file_path, "r") as file:
        return json.load(file)


def group_by_crs(patch_list: PatchList) -> Dict[CRS, List[str]]:
    patches_by_crs: DefaultDict[CRS, List[str]] = defaultdict(list)
    for name, bbox in patch_list:
        patches_by_crs[bbox.crs].append(name)
    return patches_by_crs
