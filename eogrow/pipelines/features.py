"""Implements a pipeline to construct features for training/prediction."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from pydantic import Field

from eolearn.core import (
    CopyTask,
    EONode,
    EOWorkflow,
    FeatureType,
    LoadTask,
    MergeFeatureTask,
    OverwritePermission,
    SaveTask,
)
from eolearn.core.types import Feature
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.mask import JoinMasksTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..tasks.features import (
    MaxNDVIMosaickingTask,
    MedianMosaickingTask,
    MosaickingTask,
    ValidDataFractionPredicate,
    join_valid_and_cloud_masks,
)
from ..types import PatchList, TimePeriod
from ..utils.filter import get_patches_with_missing_features
from ..utils.validators import (
    ensure_storage_key_presence,
    field_validator,
    optional_field_validator,
    parse_dtype,
    parse_time_period,
)

LOGGER = logging.getLogger(__name__)


class ValidityFiltering(BaseSchema):
    cloud_mask_feature_name: Optional[str] = Field(
        description="Name of cloud mask to enable additional filtering by cloud"
    )
    valid_data_feature_name: str = Field(description="Name of the valid-data mask to use for filtering.")

    validity_threshold: Optional[float] = Field(
        description="Threshold to remove frames with valid data lower than threshold"
    )


class FeaturesPipeline(Pipeline):
    """A pipeline to calculate and prepare features for ML"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder for the features pipeline."
        )
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the features pipeline."
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        bands_feature_name: str = Field(description="Name of data feature containing band data")

        data_preparation: ValidityFiltering

        ndis: Dict[str, Tuple[int, int]] = Field(
            default_factory=dict,
            description=(
                "A dictionary of kind `{feature_name: (id1, id2)}` that specifies how to calculate the NDIs of bands "
                "(with indices `id1` and `id2` in the bands feature) and save it under `feature_name`."
            ),
        )

        dtype: Optional[np.dtype] = Field(description="The dtype under which the concatenated features should be saved")
        _parse_dtype = optional_field_validator("dtype", parse_dtype, pre=True)
        output_feature_name: str = Field(description="Name of output data feature encompassing bands and NDIs")

    config: Schema

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of specified output features"""
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
            check_timestamps=True,
        )

    def _get_output_features(self) -> list[Feature]:
        """Lists all features that are to be saved upon the pipeline completion"""
        return [(FeatureType.DATA, self.config.output_feature_name)]

    def _get_bands_feature(self) -> Feature:
        return FeatureType.DATA, self.config.bands_feature_name

    def _get_valid_data_feature(self) -> Feature:
        return FeatureType.MASK, self.config.data_preparation.valid_data_feature_name

    def build_workflow(self) -> EOWorkflow:
        """
        Creates a workflow:
        1. Loads and prepares a 'bands_feature' and 'valid_data_feature'
        2. Temporally regularizes bands and NDIs
        3. Calculates NDIs based on 'bands_feature'
        4. Applies post-processing, which prepares all output features
        5. Saves all relevant features (specified in _get_output_features)
        """
        preparation_node = self.get_data_preparation_node()
        regularization_node = self.get_temporal_regularization_node(preparation_node)
        ndi_node = self.get_ndi_node(regularization_node)
        postprocessing_node = self.get_postprocessing_node(ndi_node)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            features=self._get_output_features(),
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            use_zarr=self.storage.config.use_zarr,
        )
        save_node = EONode(save_task, inputs=[postprocessing_node])

        return EOWorkflow.from_endnodes(save_node)

    def get_data_preparation_node(self) -> EONode:
        """Nodes that load, filter, and prepare a feature containing all bands

        :return: A node with preparation tasks and feature for masking invalid data
        """
        filtering_config = self.config.data_preparation

        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key),
            filesystem=self.storage.filesystem,
            lazy_loading=True,
        )
        end_node = EONode(load_task)

        if filtering_config.cloud_mask_feature_name:
            zip_masks_task = JoinMasksTask(
                [
                    self._get_valid_data_feature(),
                    (FeatureType.MASK, filtering_config.cloud_mask_feature_name),
                ],
                self._get_valid_data_feature(),
                join_operation=join_valid_and_cloud_masks,
            )
            end_node = EONode(zip_masks_task, inputs=[end_node], name="Combine validity mask and cloud mask")

        if filtering_config.validity_threshold is not None:
            filter_func = ValidDataFractionPredicate(filtering_config.validity_threshold)
            filter_task = SimpleFilterTask(self._get_valid_data_feature(), filter_func)
            end_node = EONode(filter_task, inputs=[end_node])

        return end_node

    def get_temporal_regularization_node(self, previous_node: EONode) -> EONode:
        """Builds node adding temporal regularization to workflow."""
        return previous_node

    def get_ndi_node(self, previous_node: EONode) -> EONode:
        """Builds a node for constructing Normalized Difference Indices"""

        for name, (id1, id2) in self.config.ndis.items():
            ndi_task = NormalizedDifferenceIndexTask(self._get_bands_feature(), (FeatureType.DATA, name), (id1, id2))
            previous_node = EONode(ndi_task, inputs=[previous_node])

        return previous_node

    def get_postprocessing_node(self, previous_node: EONode) -> EONode:
        """Tasks performed after temporal regularization. Should also prepare features for the saving step"""
        ndi_features = [(FeatureType.DATA, name) for name in self.config.ndis]
        merge_task = MergeFeatureTask(
            [self._get_bands_feature(), *ndi_features],
            (FeatureType.DATA, self.config.output_feature_name),
            dtype=self.config.dtype,
        )
        return EONode(merge_task, inputs=[previous_node])


class MosaickingSpecifications(BaseSchema):
    time_period: TimePeriod
    _parse_time_period = field_validator("time_period", parse_time_period, pre=True)
    n_mosaics: int

    max_ndi_indices: Optional[Tuple[int, int]] = Field(
        description=(
            "When omitted uses median value mosaicking. If set, uses max NDI mosaicking for the NDI of the bands at"
            " specified indices. For example, to use max NDVI when using all 13 bands of L1C set parameter to `[7, 3]`"
            " (uses B08 and B04)"
        )
    )


class MosaickingFeaturesPipeline(FeaturesPipeline):
    """A pipeline to calculate and prepare features for ML including mosaicking"""

    _NDI_FEATURE = FeatureType.DATA, "_NDI_FEATURE_OF_MOSAICKING_PIPELINE"

    class Schema(FeaturesPipeline.Schema):
        mosaicking: MosaickingSpecifications = Field(
            description=(
                "Fine-tuning of mosaicking parameters. If not set, the interpolation will work on current timestamps"
            )
        )

    config: Schema

    def get_data_preparation_node(self) -> EONode:
        preparation_node = super().get_data_preparation_node()

        if self.config.mosaicking.max_ndi_indices:
            ndi_task = NormalizedDifferenceIndexTask(
                self._get_bands_feature(), self._NDI_FEATURE, self.config.mosaicking.max_ndi_indices
            )
            return EONode(ndi_task, inputs=[preparation_node])

        return preparation_node

    def get_temporal_regularization_node(self, previous_node: EONode) -> EONode:
        start_date, end_date = self.config.mosaicking.time_period

        mosaicking_task: MosaickingTask
        if self.config.mosaicking.max_ndi_indices:
            mosaicking_task = MaxNDVIMosaickingTask(
                self._get_bands_feature(),
                (start_date, end_date, self.config.mosaicking.n_mosaics),
                self._NDI_FEATURE,
                self._get_valid_data_feature(),
            )
        else:
            mosaicking_task = MedianMosaickingTask(
                self._get_bands_feature(),
                (start_date, end_date, self.config.mosaicking.n_mosaics),
                self._get_valid_data_feature(),
            )
        mosaicking_node = EONode(mosaicking_task, inputs=[previous_node])
        return EONode(
            CopyTask(features=[self._get_bands_feature()]),
            inputs=[mosaicking_node],
            name="Remove non-mosaicked features",
        )
