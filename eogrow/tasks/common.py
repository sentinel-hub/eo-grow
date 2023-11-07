"""Common tasks shared between pipelines."""

from __future__ import annotations

from functools import partial
from typing import Any

import cv2
import numpy as np

from eolearn.core import EOPatch, EOTask, MapFeatureTask, SaveTask
from eolearn.core.types import Feature, FeaturesSpecification
from eolearn.core.utils.parsing import parse_renamed_feature
from eolearn.geometry import MorphologicalOperations


class ClassFilterTask(EOTask):
    """Run class specific morphological operation."""

    def __init__(
        self,
        feature: Feature,
        labels: list[int],
        morph_operation: MorphologicalOperations,
        struct_elem: np.ndarray | None = None,
    ):
        """Perform a morphological operation on a given feature mask

        :param feature: Feature to be modified
        :param labels: List of labels to be considered for morphological operation
        :param morph_operation: Type of morphological operation ot perform
        :param struct_elem: Structured element to be used. Taken from `ml_tools.MorphologicalStructFactory`
        """
        self.feature_name: str | None
        self.new_feature_name: str | None
        self.renamed_feature = parse_renamed_feature(feature)
        self.labels = labels

        self.morph_operation = MorphologicalOperations.get_operation(morph_operation)
        self.struct_elem = struct_elem

    def execute(self, eopatch: EOPatch) -> EOPatch:
        feature_type, feature_name, new_feature_name = self.renamed_feature
        mask = eopatch[(feature_type, feature_name)].copy()
        morp_func = partial(cv2.morphologyEx, kernel=self.struct_elem, op=self.morph_operation)

        for label in self.labels:
            label_mask = np.squeeze((mask == label).astype(np.uint8), axis=-1)
            mask_mod = morp_func(label_mask) * label
            mask_mod = mask_mod[..., np.newaxis]
            mask[mask == label] = mask_mod[mask == label]

        eopatch[(feature_type, new_feature_name)] = mask
        return eopatch


class LinearFunctionTask(MapFeatureTask):
    """Applies a linear function to the values of input features.

    Each value in the feature is modified as `x -> x * slope + intercept`. The `dtype` of the result can be customized.
    """

    def __init__(
        self,
        input_features: FeaturesSpecification,
        output_features: FeaturesSpecification | None = None,
        slope: float = 1,
        intercept: float = 0,
        dtype: str | type | np.dtype | None = None,
    ):
        """
        :param input_features: Feature or features on which the function is used.
        :param output_features: Feature or features for saving the result. If not provided the input_features are
            overwritten.
        :param slope: Slope of the function i.e. the multiplication factor.
        :param intercept: Intercept of the function i.e. the value added.
        :param dtype: Numpy dtype of the output feature. If not provided the dtype is determined by Numpy, so it is
            recommended to set manually.
        """
        if output_features is None:
            output_features = input_features
        self.dtype = dtype if dtype is None else np.dtype(dtype)

        super().__init__(input_features, output_features, slope=slope, intercept=intercept)

    def map_method(self, feature: np.ndarray, slope: float, intercept: float) -> np.ndarray:  # type:ignore[override]
        """A method where feature is multiplied by a slope"""
        rescaled_feature = feature * slope + intercept
        return rescaled_feature if self.dtype is None else rescaled_feature.astype(self.dtype)


class SkippableSaveTask(SaveTask):
    """Same as `SaveTask` but can be skipped if the `eopatch_folder` is set to `None`."""

    def execute(self, eopatch: EOPatch, *, eopatch_folder: str | None = "", **kwargs: Any) -> EOPatch:
        if eopatch_folder is None:
            return eopatch
        return super().execute(eopatch, eopatch_folder=eopatch_folder, **kwargs)
