"""Common tasks shared between pipelines."""
from typing import Callable, List, Optional, Union

import numpy as np

from eolearn.core import EOPatch, EOTask, SaveTask
from eolearn.core.utils.parsing import parse_renamed_feature
from eolearn.geometry import MorphologicalOperations

from ..types import Feature


class ClassFilterTask(EOTask):
    """Run class specific morphological operation."""

    def __init__(
        self,
        feature: Feature,
        labels: List[int],
        morph_operation: Union[MorphologicalOperations, Callable],
        struct_elem: Optional[np.ndarray] = None,
    ):
        """Perform a morphological operation on a given feature mask

        :param feature: Feature to be modified
        :param labels: List of labels to be considered for morphological operation
        :param morph_operation: Type of morphological operation ot perform
        :param struct_elem: Structured element to be used. Taken from `ml_tools.MorphologicalStructFactory`
        """
        self.feature_name: Optional[str]
        self.new_feature_name: Optional[str]
        self.renamed_feature = parse_renamed_feature(feature)
        self.labels = labels

        if isinstance(morph_operation, MorphologicalOperations):
            self.morph_operation = MorphologicalOperations.get_operation(morph_operation)
        else:
            self.morph_operation = morph_operation
        self.struct_elem = struct_elem

    def execute(self, eopatch: EOPatch) -> EOPatch:
        feature_type, feature_name, new_feature_name = self.renamed_feature
        mask = eopatch[(feature_type, feature_name)].copy()

        for label in self.labels:
            label_mask = np.squeeze((mask == label), axis=-1)
            mask_mod = self.morph_operation(label_mask, self.struct_elem) * label
            mask_mod = mask_mod[..., np.newaxis]
            mask[mask == label] = mask_mod[mask == label]

        eopatch[(feature_type, new_feature_name)] = mask
        return eopatch


class SkippableSaveTask(SaveTask):
    """Same as `SaveTask` but can be skipped if the `eopatch_folder` is set to `None`."""

    def execute(self, eopatch: EOPatch, *, eopatch_folder: Optional[str] = "") -> EOPatch:
        if eopatch_folder is None:
            return eopatch
        return super().execute(eopatch, eopatch_folder=eopatch_folder)
