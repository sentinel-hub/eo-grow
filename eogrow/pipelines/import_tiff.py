"""Implements a pipeline for importing reference data from a raster image."""

from __future__ import annotations

from typing import Optional

import fs
import numpy as np
from pydantic import Field

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, OverwritePermission, SaveTask
from eolearn.core.types import Feature
from eolearn.features.feature_manipulation import SpatialResizeTask
from eolearn.features.utils import ResizeLib, ResizeMethod, ResizeParam
from eolearn.io import ImportFromTiffTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..types import PatchList
from ..utils.filter import get_patches_with_missing_features
from ..utils.validators import ensure_storage_key_presence, optional_field_validator, parse_dtype


class ResizeSchema(BaseSchema):
    """How to resize the tiff data after adding it to EOPatches."""

    resize_type: ResizeParam = Field(
        description=(
            "Determines type of resizing process and how `width_param` and `height_param` are used. See"
            " `SpatialResizeTask` documentation for more info."
        )
    )
    width_param: float = Field(description="Parameter to be applied to the width in combination with the resize_type.")
    height_param: float = Field(
        description="Parameter to be applied to the height in combination with the resize_type."
    )
    method: ResizeMethod = ResizeMethod.LINEAR
    library: ResizeLib = ResizeLib.CV2


class ImportTiffPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(description="The storage manager key of the output folder.")
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        tiff_folder_key: str = Field(
            "input_data",
            description="The storage manager key of the folder containing the tiff. Defaults to the input-data folder.",
        )
        _ensure_tiff_folder_key = ensure_storage_key_presence("tiff_folder_key")
        input_filename: str = Field(description="Name of tiff file to import.")
        output_feature: Feature = Field(description="Feature containing the imported tiff information.")
        no_data_value: float = Field(
            np.nan, description="Value assigned to undefined pixels, e.g. outside of given input image."
        )
        dtype: Optional[np.dtype] = Field(description="Custom dtype for the imported feature.")
        _parse_dtype = optional_field_validator("dtype", parse_dtype, pre=True)
        resize: Optional[ResizeSchema] = Field(
            description="Settings for SpatialResizeTask applied at the end. When omitted resizing is not performed."
        )

    config: Schema

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of new features."""
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            [self.config.output_feature],
            check_timestamps=False,
        )

    def build_workflow(self) -> EOWorkflow:
        create_eopatch_node = EONode(CreateEOPatchTask())

        file_path = fs.path.join(self.storage.get_folder(self.config.tiff_folder_key), self.config.input_filename)
        import_task = ImportFromTiffTask(
            self.config.output_feature,
            file_path,
            no_data_value=self.config.no_data_value,
            filesystem=self.storage.filesystem,
            image_dtype=self.config.dtype,
        )
        import_node = EONode(import_task, inputs=[create_eopatch_node])

        resize_node = None
        if self.config.resize:
            width_param, height_param = self.config.resize.width_param, self.config.resize.height_param
            if self.config.resize.resize_type is ResizeParam.NEW_SIZE:
                # pydantic transforms input to floats, but the function fails unles integers are provided for NEW_SIZE
                width_param, height_param = round(width_param), round(height_param)

            resize_task = SpatialResizeTask(
                resize_type=self.config.resize.resize_type,
                height_param=height_param,
                width_param=width_param,
                resize_method=self.config.resize.method,
                resize_library=self.config.resize.library,
            )
            resize_node = EONode(resize_task, inputs=[import_node])

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
            features=[self.config.output_feature],
            use_zarr=self.storage.config.use_zarr,
        )
        save_node = EONode(save_task, inputs=[resize_node or import_node])

        return EOWorkflow.from_endnodes(save_node)
