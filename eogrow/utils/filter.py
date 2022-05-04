"""
Utilities for filtering eopatch lists
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

import fs
from fs.base import FS
from tqdm.auto import tqdm

from eolearn.core.eodata_io import walk_filesystem

from ..utils.types import FeatureSpec


def check_if_features_exist(
    filesystem: FS,
    eopatch_path: str,
    features: Iterable[FeatureSpec],
) -> bool:
    """Checks whether an EOPatch in the given location has all specified features saved"""
    not_seen_features = set(features)
    try:
        for ftype, name, _ in walk_filesystem(filesystem, eopatch_path, features=features):
            if (ftype, name) in not_seen_features:
                not_seen_features.remove((ftype, name))
            elif ftype in not_seen_features:
                not_seen_features.remove(ftype)
    except (IOError, fs.errors.ResourceNotFound):
        return False
    return len(not_seen_features) == 0


def get_patches_with_missing_features(
    filesystem: FS,
    patches_folder: str,
    patch_list: List[str],
    features: Iterable[FeatureSpec],
) -> List[str]:
    """Filters out names of those EOPatches that are missing some given features.

    :param filesystem: A filesystem object.
    :param patches_folder: A path to folder with EOPatches, relative to `filesystem` object.
    :param patch_list: A list of EOPatch names.
    :param features: A list of EOPatch features.
    :return: A sublist of `patch_list` with only EOPatch names that are missing some features.
    """
    eopatch_paths = [fs.path.combine(patches_folder, eopatch) for eopatch in patch_list]

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
