"""Implements a pipeline for rasterizing vector datasets."""
import logging
import os
import uuid
from typing import Any, List, Optional, Tuple, Union

import fiona
import fs
import geopandas as gpd
import numpy as np
from pydantic import Field

from eolearn.core import CreateEOPatchTask, EONode, EOWorkflow, FeatureType, LoadTask, OverwritePermission, SaveTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import VectorImportTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..types import Feature, FeatureSpec, PatchList
from ..utils.filter import get_patches_with_missing_features
from ..utils.fs import LocalFile
from ..utils.validators import ensure_exactly_one_defined, field_validator, parse_dtype, restrict_types
from ..utils.vector import concat_gdf

LOGGER = logging.getLogger(__name__)


class Preprocessing(BaseSchema):
    reproject_crs: Optional[int] = Field(
        description=(
            "An EPSG code of a CRS in which vector_input data will be reprojected once loaded. This parameter "
            "is mandatory if vector_input contains multiple layers in different CRS"
        ),
    )


class RasterizePipeline(Pipeline):
    """A pipeline module for rasterizing vector datasets."""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder for the rasterization pipeline."
        )
        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the rasterization pipeline pipeline."
        )
        vector_input: Union[Feature, str] = Field(
            description="A filename in the input_data folder or a feature containing vector data."
        )
        dataset_layer: Optional[str] = Field(
            description="Name of a layer with data to be rasterized in a multi-layer file."
        )
        raster_value: Optional[float] = Field(
            description="Value to use for all rasterized polygons. Use either this or `values_column`."
        )
        raster_values_column: Optional[str] = Field(
            description=(
                "GeoPandas dataframe column name from which to read values for geometries. Use either this or `value`."
            )
        )
        raster_feature: Feature = Field(description="Output feature of rasterization.")
        polygon_buffer: float = Field(
            0, description="The size of polygon buffering to be applied before rasterization."
        )
        resolution: Optional[float] = Field(
            description="Rendering resolution in meters. Cannot be used with `raster_shape`."
        )
        raster_shape: Optional[Tuple[int, int]] = Field(
            description="Shape of resulting raster image. Cannot be used with `resolution`."
        )
        overlap_value: Optional[int] = Field(description="Value to write over the areas where polygons overlap.")
        dtype: np.dtype = Field(np.dtype("int32"), description="Numpy dtype of the output feature.")
        no_data_value: int = Field(0, description="The no_data_value argument to be passed to VectorToRasterTask")

        preprocess_dataset: Optional[Preprocessing] = Field(
            description=(
                "Parameters used by self.preprocess_dataset method. If set to `None` it skips the dataframe preprocess"
                " step."
            )
        )
        compress_level: int = Field(1, description="Level of compression used in saving EOPatches")

        _parse_dtype = field_validator("dtype", parse_dtype, pre=True)
        _check_raster_value_values_column = ensure_exactly_one_defined("raster_value", "raster_values_column")
        _check_shape_resolution = ensure_exactly_one_defined("raster_shape", "resolution")
        _check_raster_feature_type = field_validator("raster_feature", restrict_types(lambda ftype: ftype.is_image()))

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.filename: Optional[str] = None

        output_is_temporal = self._is_temporal(self.config.raster_feature)
        if isinstance(self.config.vector_input, str):
            self.filename = self._parse_input_file(self.config.vector_input)
            feature_type = FeatureType.VECTOR if output_is_temporal else FeatureType.VECTOR_TIMELESS
            self.vector_feature = feature_type, f"TEMP_{uuid.uuid4().hex}"
        else:
            if output_is_temporal != self._is_temporal(self.config.vector_input):
                raise ValueError(
                    "The requested output feature does not correspond to input vector feature."
                    " Both input and output should be the same, either temporal or timeless."
                )
            self.vector_feature = self.config.vector_input

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
        )

        return filtered_patch_list

    def run_procedure(self) -> Tuple[List[str], List[str]]:
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

        dataset_gdf = concat_gdf(dataset_layers, reproject_crs=preprocess_config.reproject_crs)

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
                self.vector_feature,
                path=path,
                filesystem=self.storage.filesystem,
                layer=self.config.dataset_layer,
            )
            data_preparation_node = EONode(import_task, inputs=[create_node])
        else:
            features = [self.vector_feature, FeatureType.BBOX]
            if self._is_temporal(self.vector_feature):
                features.append(FeatureType.TIMESTAMPS)
            input_task = LoadTask(
                self.storage.get_folder(self.config.input_folder_key),
                filesystem=self.storage.filesystem,
                features=features,
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
            compress_level=self.config.compress_level,
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
                raster_feature=self.config.raster_feature,
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

    @staticmethod
    def _parse_input_file(value: str) -> str:
        """Checks if given name ends with one of the supported file extensions"""
        if not value.lower().endswith((".geojson", ".shp", ".gpkg", ".gdb")):
            raise ValueError(f"Input file path {value} should be a GeoJSON, Shapefile, GeoPackage or GeoDataBase.")
        return value

    def _get_dataset_path(self, filename: str) -> str:
        """Provides a path from where dataset should be loaded into the workflow"""
        if self.config.preprocess_dataset is not None:
            folder = self.storage.get_cache_folder()
            filename = f"preprocessed_{filename}"
            filename = (os.path.splitext(filename))[0] + ".gpkg"
        else:
            folder = self.storage.get_input_data_folder()

        return fs.path.combine(folder, filename)

    def _get_output_features(self) -> List[FeatureSpec]:
        """Lists all features that are to be saved upon the pipeline completion"""
        features: List[FeatureSpec] = [self.config.raster_feature, FeatureType.BBOX]

        if self._is_temporal(self.vector_feature):
            features.append(FeatureType.TIMESTAMPS)

        return features

    @staticmethod
    def _is_temporal(feature: Feature) -> bool:
        f_type, _ = feature
        return f_type.is_temporal()
