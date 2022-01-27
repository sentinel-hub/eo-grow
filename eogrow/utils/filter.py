"""
Utilities for filtering eopatch lists
"""
from typing import List, Iterable
from concurrent.futures import ThreadPoolExecutor

import fs
from fs.base import FS

from eolearn.core.eodata_io import walk_filesystem

from ..utils.types import Feature


def check_if_features_exist(
    filesystem: FS,
    eopatch_path: str,
    features: Iterable[Feature],
) -> bool:
    """Checks whether an EOPatch in the given location has all specified features saved"""
    not_seen_features = set(features)
    try:
        for (ftype, name, _) in walk_filesystem(filesystem, eopatch_path, features=features):
            if (ftype, name) in not_seen_features:
                not_seen_features.remove((ftype, name))
            elif ftype in not_seen_features:
                not_seen_features.remove(ftype)
    except (IOError, fs.errors.ResourceNotFound):
        return False
    return len(not_seen_features) == 0


def get_patches_without_all_features(
    filesystem: FS,
    patches_folder: str,
    patch_list: List[str],
    features: Iterable[Feature],
) -> List[str]:
    """Filters out patches that already contain all features"""
    eopatch_paths = [fs.path.combine(patches_folder, eopatch) for eopatch in patch_list]

    def check_patch(eopatch_path):
        return check_if_features_exist(filesystem, eopatch_path, features)

    with ThreadPoolExecutor() as pool:
        has_features_iter = pool.map(check_patch, eopatch_paths)

    return [eopatch for eopatch, has_features in zip(patch_list, has_features_iter) if not has_features]
