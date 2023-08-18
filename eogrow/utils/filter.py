"""
Utilities for filtering eopatch lists
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import fs
from fs.base import FS
from tqdm.auto import tqdm

from eolearn.core import FeatureType
from eolearn.core.eodata_io import get_filesystem_data_info

from ..types import PatchList


def check_if_features_exist(
    filesystem: FS,
    eopatch_path: str,
    features: Sequence[tuple[FeatureType, str]],
    skip_bbox: bool = False,
    skip_timestamps: bool = False,
) -> bool:
    """Checks whether an EOPatch in the given location has all specified features saved"""
    try:
        existing_data = get_filesystem_data_info(filesystem, eopatch_path, features)
    except (IOError, fs.errors.ResourceNotFound):
        return False

    if not skip_bbox and existing_data.bbox is None:
        return False
    if not skip_timestamps and existing_data.timestamps is None:
        return False

    return all(fname in existing_data.features.get(ftype, []) for ftype, fname in features)


def get_patches_with_missing_features(
    filesystem: FS,
    patches_folder: str,
    patch_list: PatchList,
    features: Sequence[tuple[FeatureType, str]],
) -> PatchList:
    """Filters out names of those EOPatches that are missing some given features.

    :param filesystem: A filesystem object.
    :param patches_folder: A path to folder with EOPatches, relative to `filesystem` object.
    :param patch_list: A list of EOPatch names.
    :param features: A list of EOPatch features.
    :return: A sublist of `patch_list` with only EOPatch names that are missing some features.
    """
    eopatch_paths = [fs.path.combine(patches_folder, eopatch) for eopatch, _ in patch_list]

    def check_patch(eopatch_path: str) -> bool:
        return check_if_features_exist(filesystem, eopatch_path, features)

    with ThreadPoolExecutor() as executor:
        has_features_list = list(
            tqdm(
                executor.map(check_patch, eopatch_paths),
                total=len(eopatch_paths),
                desc="Checking EOPatches",
            )
        )

    return [eopatch for eopatch, has_features in zip(patch_list, has_features_list) if not has_features]
