"""
Common tasks shared between pipelines
"""
from typing import Callable, List, Optional, Union

import numpy as np

from eolearn.core import EOPatch, EOTask, MapFeatureTask
from eolearn.geometry import MorphologicalOperations

from ..utils.types import Feature


class MappingTask(MapFeatureTask):
    """A task that takes an input mask_timeless feature and creates an output mask_timeless feature, where the
    input mask values are mapped to the output mask values, using the provided mapping dictionary.
    """

    def __init__(self, input_feature: Feature, output_feature: Feature, mapping_dict: dict):
        super().__init__(input_feature, output_feature, mapping_dict=mapping_dict)

    def map_method(self, feature: np.ndarray, mapping_dict: dict) -> np.ndarray:
        mapped_values = feature.copy()
        for map_from, map_to in mapping_dict.items():
            mapped_values[feature == map_from] = map_to

        return mapped_values


class ClassFilterTask(EOTask):
    """
    Run class specific morphological operation.
    """

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
        self.feature_type, self.feature_name, self.new_feature_name = self.parse_renamed_feature(feature)
        self.labels = labels

        if isinstance(morph_operation, MorphologicalOperations):
            self.morph_operation = MorphologicalOperations.get_operation(morph_operation)
        else:
            self.morph_operation = morph_operation
        self.struct_elem = struct_elem

    def execute(self, eopatch: EOPatch) -> EOPatch:
        mask = eopatch[self.feature_type][self.feature_name].copy()

        for label in self.labels:
            label_mask = np.squeeze((mask == label), axis=-1)
            mask_mod = self.morph_operation(label_mask, self.struct_elem) * label
            mask_mod = mask_mod[..., np.newaxis]
            mask[mask == label] = mask_mod[mask == label]

        eopatch[self.feature_type][self.new_feature_name] = mask
        return eopatch
