"""Implements a pipeline for rasterizing vector datasets."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional, Tuple, Union

import fiona
import fs
import geopandas as gpd
import numpy as np
from pydantic import Field, validator

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, FeatureType, LoadTask, OverwritePermission, SaveTask
from eolearn.core.types import Feature
from eolearn.geometry import VectorToRasterTask
from eolearn.io import VectorImportTask
from sentinelhub import CRS

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..types import PatchList
from ..utils.filter import get_patches_with_missing_features
from ..utils.fs import LocalFile
from ..utils.validators import ensure_exactly_one_defined, ensure_storage_key_presence, field_validator, parse_dtype
from ..utils.vector import concat_gdf

LOGGER = logging.getLogger(__name__)


class Preprocessing(BaseSchema):
    reproject_crs: Optional[int] = Field(
        description=(
            "An EPSG code of a CRS in which vector_input data will be reprojected once loaded. Mandatory if"
            " the input vector contains multiple layers in different CRS."
        ),
    )


class RasterizePipeline(Pipeline):
    """A pipeline module for rasterizing vector datasets."""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(description="The storage manager key pointing to the input folder.")
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        output_folder_key: str = Field(description="The storage manager key pointing to the output folder.")
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        vector_input: Union[Feature, str] = Field(description="An input filename or a feature containing vector data.")
        output_feature: Feature = Field(description="A feature which should contain the newly rasterized data.")

        raster_value: Optional[float] = Field(
            description="Value to be used for all rasterized polygons. Cannot be used with `raster_values_column`."
        )
        raster_values_column: Optional[str] = Field(
            description=(
                "GeoPandas column for reading per-geometry rasterization values. Cannot be used with `raster_value`."
            )
        )
        _check_raster_value_values_column = ensure_exactly_one_defined("raster_value", "raster_values_column")

        resolution: Optional[float] = Field(
            description="Rendering resolution in meters. Cannot be used with `raster_shape`."
        )
        raster_shape: Optional[Tuple[int, int]] = Field(
            description="Shape of resulting raster image. Cannot be used with `resolution`."
        )
        _check_shape_resolution = ensure_exactly_one_defined("resolution", "raster_shape")

        dtype: np.dtype = Field(np.dtype("int32"), description="Numpy dtype of the output feature.")
        _parse_dtype = field_validator("dtype", parse_dtype, pre=True)

        preprocess_dataset: Optional[Preprocessing] = Field(
            description="Parameters used by `self.preprocess_dataset` method. Skipped if set to `None`."
        )
        dataset_layer: Optional[str] = Field(
            description="In case of multi-layer files, name of the layer with data to be rasterized."
        )
        polygon_buffer: float = Field(
            0, description="The size of polygon buffering to be applied before rasterization."
        )
        overlap_value: Optional[float] = Field(description="Value to write over the areas where polygons overlap.")
        no_data_value: int = Field(0, description="The no_data_value argument to be passed to VectorToRasterTask")

        @validator("vector_input")
        def _check_vector_input(cls, vector_input: Feature | str) -> Feature | str:
            if isinstance(vector_input, str):
                assert vector_input.lower().endswith(
                    (".geojson", ".shp", ".gpkg", ".gdb")
                ), f"Input file path {vector_input} should be a GeoJSON, Shapefile, GeoPackage or GeoDataBase."
            else:
                assert vector_input[0].is_vector(), "Only vector-like input features are allowed!"
            return vector_input

        @validator("output_feature")
        def _check_temporal_nature_match(cls, output_feature: Feature, values: Dict[str, Any]) -> Feature:
            assert output_feature[0].is_image(), "Only image-like output features are allowed!"

            vector_input = values.get("vector_input")
            if vector_input and not isinstance(vector_input, str):
                assert (
                    vector_input[0].is_temporal() == output_feature[0].is_temporal()
                ), "Temporal natures of the vector input and the raster output must match!"
            return output_feature

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.filename: str | None = None
        self.vector_feature: Feature

        if not isinstance(self.config.vector_input, str):
            self.vector_feature = self.config.vector_input
        else:
            ftype = FeatureType.VECTOR if self.config.output_feature[0].is_temporal() else FeatureType.VECTOR_TIMELESS
            self.filename = self.config.vector_input
            self.vector_feature = (ftype, f"TEMP_{uuid.uuid4().hex}")

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        output_features = self._get_output_features()
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            output_features,
            check_timestamps=any(ftype.is_temporal() for ftype, _ in output_features),
        )

    def run_procedure(self) -> tuple[list[str], list[str]]:
        if self.filename is not None and self.config.preprocess_dataset is not None:
            self.run_dataset_preprocessing(self.filename, self.config.preprocess_dataset)

        return super().run_procedure()

    def run_dataset_preprocessing(self, filename: str, preprocess_config: Preprocessing) -> None:
        """Loads datasets, applies preprocessing steps and saves them to a cache folder"""
        LOGGER.info("Preprocessing dataset %s", filename)
        gpd_engine = self.storage.config.geopandas_backend

        file_path = fs.path.combine(self.storage.get_input_data_folder(), filename)
        with LocalFile(file_path, mode="r", filesystem=self.storage.filesystem) as local_file:
            dataset_layers = [
                gpd.read_file(local_file.path, layer=layer, encoding="utf-8", engine=gpd_engine)
                for layer in fiona.listlayers(local_file.path)
            ]

        reproject_crs = None if preprocess_config.reproject_crs is None else CRS(preprocess_config.reproject_crs)
        dataset_gdf = concat_gdf(dataset_layers, reproject_crs=reproject_crs)
        dataset_gdf = self.preprocess_dataset(dataset_gdf)
        dataset_path = self._get_dataset_path(filename)

        with LocalFile(dataset_path, mode="w", filesystem=self.storage.filesystem) as local_file:
            dataset_gdf.to_file(local_file.path, encoding="utf-8", driver="GPKG", engine=gpd_engine)

    def build_workflow(self) -> EOWorkflow:
        """Creates workflow that is divided into the following sub-parts:

        1. loading data,
        2. preprocessing steps,
        3. rasterization of features,
        4. postprocessing steps,
        5. saving results
        """
        if self.filename is not None:
            create_node = EONode(CreateEOPatchTask())
            path = self._get_dataset_path(self.filename)
            import_task = VectorImportTask(
                feature=self.vector_feature,
                path=path,
                filesystem=self.storage.filesystem,
                layer=self.config.dataset_layer,
            )
            data_preparation_node = EONode(import_task, inputs=[create_node])
        else:
            input_task = LoadTask(
                self.storage.get_folder(self.config.input_folder_key),
                filesystem=self.storage.filesystem,
                features=[self.vector_feature],
            )
            data_preparation_node = EONode(input_task)

        preprocess_node = self.get_prerasterization_node(data_preparation_node)
        rasterization_node = self.get_rasterization_node(preprocess_node)
        postprocess_node = self.get_postrasterization_node(rasterization_node)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            features=self._get_output_features(),
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            use_zarr=self.storage.config.use_zarr,
        )
        save_node = EONode(save_task, inputs=[postprocess_node])

        return EOWorkflow.from_endnodes(save_node)

    def preprocess_dataset(self, dataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:  # pylint: disable=no-self-use
        """Method for applying custom preprocessing steps on the entire dataset"""
        return dataframe

    def get_prerasterization_node(self, previous_node: EONode) -> EONode:  # pylint: disable=no-self-use
        """Builds node with tasks to be applied after loading vector feature but before rasterization"""
        return previous_node

    def get_rasterization_node(self, previous_node: EONode) -> EONode:
        """Builds nodes containing rasterization tasks"""

        return EONode(
            inputs=[previous_node],
            task=VectorToRasterTask(
                vector_input=self.vector_feature,
                raster_feature=self.config.output_feature,
                values_column=self.config.raster_values_column,
                values=self.config.raster_value,
                buffer=self.config.polygon_buffer,
                raster_resolution=self.config.resolution,
                raster_shape=self.config.raster_shape,
                raster_dtype=np.dtype(self.config.dtype),
                no_data_value=self.config.no_data_value,
                overlap_value=self.config.overlap_value,
            ),
        )

    def get_postrasterization_node(self, previous_node: EONode) -> EONode:  # pylint: disable=no-self-use
        """Builds node with tasks to be applied after rasterization"""
        return previous_node

    def _get_dataset_path(self, filename: str) -> str:
        """Provides a path from where dataset should be loaded into the workflow"""
        if self.config.preprocess_dataset is not None:
            folder = self.storage.get_cache_folder()
            filename = f"preprocessed_{filename}"
            filename = (os.path.splitext(filename))[0] + ".gpkg"
        else:
            folder = self.storage.get_input_data_folder()

        return fs.path.combine(folder, filename)

    def _get_output_features(self) -> list[Feature]:
        """Lists all features that are to be saved upon the pipeline completion"""
        return [self.config.output_feature]
