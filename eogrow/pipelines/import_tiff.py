"""Implements a pipeline for importing reference data from a raster image."""
from typing import List, Optional

import fs
import numpy as np
from pydantic import Field

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.features.feature_manipulation import SpatialResizeTask
from eolearn.features.utils import ResizeLib, ResizeMethod
from eolearn.io import ImportFromTiffTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..utils.filter import get_patches_with_missing_features
from ..utils.types import Feature
from ..utils.validators import optional_field_validator, parse_dtype


class ResizeSchema(BaseSchema):
    """Uses resolution instead of new_size or scale_factors."""

    source_resolution: int = Field(description="Resolution of .tiff file.")
    target_resolution: int = Field(description="Resolution of EOPatch.")
    resize_method: ResizeMethod = ResizeMethod.LINEAR
    resize_library: ResizeLib = ResizeLib.PIL


class ImportTiffPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        output_folder_key: str
        input_filename: str
        output_feature: Feature
        no_data_value: float = Field(
            np.nan, description="Value assigned to undefined pixels, e.g. outside of given input image"
        )
        dtype: Optional[np.dtype] = Field(description="Custom dtype for the imported feature.")
        _parse_dtype = optional_field_validator("dtype", parse_dtype, pre=True)
        use_vsi: bool = Field(
            True,
            description="Whether to use the VSI for reading. Enabled by default as a remote filesystem is assumed.",
        )
        resize: Optional[ResizeSchema] = Field(
            description="Settings for SpatialResizeTask applied at the end. When omitted resizing is not performed."
        )

    config: Schema

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """EOPatches are filtered according to existence of new features."""
        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            [self.config.output_feature, FeatureType.BBOX],
        )
        return filtered_patch_list

    def add_resize_node(self, previous_node: EONode) -> EONode:
        """Adds a resize node if requested."""
        resize = self.config.resize
        if not resize or resize.source_resolution == resize.target_resolution:
            return previous_node

        scale_factor = resize.source_resolution / resize.target_resolution
        resize_task = SpatialResizeTask(
            scale_factors=(scale_factor, scale_factor),
            resize_method=resize.resize_method,
            resize_library=resize.resize_library,
        )
        return EONode(resize_task, inputs=[previous_node])

    def build_workflow(self) -> EOWorkflow:
        create_eopatch_node = EONode(CreateEOPatchTask())

        file_path = fs.path.join(self.storage.get_input_data_folder(), self.config.input_filename)
        import_task = ImportFromTiffTask(
            self.config.output_feature,
            file_path,
            no_data_value=self.config.no_data_value,
            filesystem=self.storage.filesystem,
            image_dtype=self.config.dtype,
            use_vsi=self.config.use_vsi,
        )
        import_node = EONode(import_task, inputs=[create_eopatch_node])

        resize_node = self.add_resize_node(import_node)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            compress_level=1,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
            features=[self.config.output_feature, FeatureType.BBOX],
        )
        save_node = EONode(save_task, inputs=[resize_node])

        return EOWorkflow.from_endnodes(save_node)
