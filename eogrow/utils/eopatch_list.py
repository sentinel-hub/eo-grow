"""Utilities for working with lists of EOPatch names."""

import json
from collections import defaultdict
from typing import DefaultDict, Dict, List

from fs.base import FS

from sentinelhub import CRS

from ..types import PatchList


def save_names(filesystem: FS, file_path: str, names: List[str]) -> None:
    """Saves a list of names (EOPatch, execution, etc.) to a file

    :param filesystem: Filesystem used to save the file.
    :param filename: Path of a JSON file where names will be saved.
    :param names: A list of names.
    """

    with filesystem.open(file_path, "w") as file:
        json.dump(names, file, indent=2)


def load_names(filesystem: FS, file_path: str) -> List[str]:
    """Loads a list of names (EOPatch, execution, etc.) from a file

    :param filesystem: Filesystem used to load the file.
    :param filename: Path of a JSON file where names are saved.
    :return: A list of names loaded from file.
    """
    with filesystem.open(file_path, "r") as file:
        return json.load(file)


def group_by_crs(patch_list: PatchList) -> Dict[CRS, List[str]]:
    patches_by_crs: DefaultDict[CRS, List[str]] = defaultdict(list)
    for name, bbox in patch_list:
        patches_by_crs[bbox.crs].append(name)
    return patches_by_crs
