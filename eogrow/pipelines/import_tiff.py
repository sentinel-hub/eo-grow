"""Implements a pipeline for importing reference data from a raster image."""
from typing import List, Optional, Tuple

import fs
import numpy as np
from pydantic import Field, validator

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.features.feature_manipulation import SpatialResizeTask
from eolearn.features.utils import ResizeLib, ResizeMethod, ResizeParam
from eolearn.io import ImportFromTiffTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..utils.filter import get_patches_with_missing_features
from ..utils.types import Feature
from ..utils.validators import optional_field_validator, parse_dtype


class ResizeSchema(BaseSchema):
    """Uses resolution instead of new_size or scale_factors."""

    parameters: Tuple[ResizeParam, Tuple[float, float]] = Field(
        description=(
            "Specify the resize parameters in the same way as for `SpatialResizeTask`. Examples:"
            ' `["resolution", [10, 20]]` for changing resolution from 10 to 20,'
            ' `["new_size", [500, 1000]]` for making the imported data of size (500, 1000).'
        )
    )
    method: ResizeMethod = ResizeMethod.LINEAR
    library: ResizeLib = ResizeLib.PIL

    @validator("parameters")
    def parameters_parser(cls, v: Tuple[ResizeParam, Tuple[float, float]]) -> Tuple[ResizeParam, Tuple[float, float]]:
        kind, params = v
        if kind is ResizeParam.NEW_SIZE:
            params = (round(params[0]), round(params[1]))
        return kind, params


class ImportTiffPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        output_folder_key: str
        tiff_folder_key: str = "input_data"
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

    def build_workflow(self) -> EOWorkflow:
        create_eopatch_node = EONode(CreateEOPatchTask())

        file_path = fs.path.join(self.storage.get_folder(self.config.tiff_folder_key), self.config.input_filename)
        import_task = ImportFromTiffTask(
            self.config.output_feature,
            file_path,
            no_data_value=self.config.no_data_value,
            filesystem=self.storage.filesystem,
            image_dtype=self.config.dtype,
            use_vsi=self.config.use_vsi,
        )
        import_node = EONode(import_task, inputs=[create_eopatch_node])

        resize_node = None
        if self.config.resize:
            resize_task = SpatialResizeTask(
                resize_parameters=self.config.resize.parameters,
                resize_method=self.config.resize.method,
                resize_library=self.config.resize.library,
            )
            resize_node = EONode(resize_task, inputs=[import_node])

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            compress_level=1,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
            features=[self.config.output_feature, FeatureType.BBOX],
        )
        save_node = EONode(save_task, inputs=[resize_node or import_node])

        return EOWorkflow.from_endnodes(save_node)
