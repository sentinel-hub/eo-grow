"""
Utilities for filtering eopatch lists
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import fs
from fs.base import FS
from tqdm.auto import tqdm

from eolearn.core.eodata_io import get_filesystem_data_info
from eolearn.core.types import Feature

from ..types import PatchList


def check_if_features_exist(
    filesystem: FS,
    eopatch_path: str,
    features: Sequence[Feature],
    *,
    check_bbox: bool = True,
    check_timestamps: bool,
) -> bool:
    """Checks whether an EOPatch in the given location has all specified features saved"""
    try:
        existing_data = get_filesystem_data_info(filesystem, eopatch_path, features)
    except (IOError, fs.errors.ResourceNotFound):
        return False

    if check_bbox and existing_data.bbox is None:
        return False
    if check_timestamps and existing_data.timestamps is None:
        return False

    return all(fname in existing_data.features.get(ftype, []) for ftype, fname in features)


def get_patches_with_missing_features(
    filesystem: FS,
    patches_folder: str,
    patch_list: PatchList,
    features: Sequence[Feature],
    *,
    check_bbox: bool = True,
    check_timestamps: bool,
) -> PatchList:
    """Filters out names of those EOPatches that are missing some given features.

    :param filesystem: A filesystem object.
    :param patches_folder: A path to folder with EOPatches, relative to `filesystem` object.
    :param patch_list: A list of EOPatch names.
    :param features: A list of EOPatch features.
    :param check_bbox: Make sure that the bbox is present.
    :param check_timestamps: Make sure that the timestamps are present.
    :return: A sublist of `patch_list` with only EOPatch names that are missing some features.
    """
    eopatch_paths = [fs.path.combine(patches_folder, eopatch) for eopatch, _ in patch_list]

    def check_patch(eopatch_path: str) -> bool:
        return check_if_features_exist(
            filesystem, eopatch_path, features, check_bbox=check_bbox, check_timestamps=check_timestamps
        )

    with ThreadPoolExecutor() as executor:
        has_features_list = list(
            tqdm(
                executor.map(check_patch, eopatch_paths),
                total=len(eopatch_paths),
                desc="Checking EOPatches",
            )
        )

    return [eopatch for eopatch, has_features in zip(patch_list, has_features_list) if not has_features]
