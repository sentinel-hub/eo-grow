"""
A pipeline module for rasterizing vector datasets.
"""
import logging
import os
import uuid
from typing import Any, List, Optional, Tuple, Union

import fiona
import fs
import geopandas as gpd
import numpy as np
from pydantic import Field, validator

from eolearn.core import (
    CreateEOPatchTask,
    EONode,
    EOWorkflow,
    FeatureType,
    LoadTask,
    MergeEOPatchesTask,
    OverwritePermission,
    SaveTask,
)
from eolearn.core.utils.fs import join_path
from eolearn.geometry import VectorToRasterTask
from eolearn.io import VectorImportTask

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..utils.filter import get_patches_with_missing_features
from ..utils.fs import LocalFile
from ..utils.types import Feature, FeatureSpec
from ..utils.vector import concat_gdf

LOGGER = logging.getLogger(__name__)


class VectorColumnSchema(BaseSchema):
    """Parameter structure for individual feature / dataset column to be rasterized"""

    value: Optional[float] = Field(
        description="Value to use for all rasterized polygons. Use either this or `values_column`."
    )
    values_column: Optional[str] = Field(
        description=(
            "GeoPandas dataframe column name from which to read values for geometries. Use either this or `value`."
        )
    )
    output_feature: Feature = Field(description="Output feature of rasterization.")
    polygon_buffer: float = Field(0, description="The size of polygon buffering to be applied before rasterization.")
    resolution: float = Field(description="Rendering resolution in meters.")
    overlap_value: Optional[int] = Field(description="Value to write over the areas where polygons overlap.")
    dtype: str = Field("int32", description="Numpy dtype of the output feature.")
    no_data_value: int = Field(0, description="The no_data_value argument to be passed to VectorToRasterTask")

    @validator("values_column")
    def check_value_settings(cls, v, values):  # type: ignore
        """Ensures that precisely one of `value` and `values_column` is set."""
        assert (v is None) != (values["value"] is None), "Only one of `values_column` and `value` should be given."
        return v


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
        columns: List[VectorColumnSchema]
        preprocess_dataset: Optional[Preprocessing] = Field(
            description=(
                "Parameters used by self.preprocess_dataset method. If set to `None` it skips the dataframe preprocess"
                " step."
            )
        )

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.filename: Optional[str] = None
        if isinstance(self.config.vector_input, str):
            self.filename = self._parse_input_file(self.config.vector_input)
            self.vector_feature = FeatureType.VECTOR_TIMELESS, f"TEMP_{uuid.uuid4().hex}"
        else:
            self.vector_feature = self.config.vector_input

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
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

        file_path = fs.path.combine(self.storage.get_input_data_folder(), filename)
        with LocalFile(file_path, mode="r", filesystem=self.storage.filesystem) as local_file:
            dataset_layers = [
                gpd.read_file(local_file.path, layer=layer, encoding="utf-8")
                for layer in fiona.listlayers(local_file.path)
            ]

        dataset_gdf = concat_gdf(dataset_layers, reproject_crs=preprocess_config.reproject_crs)

        dataset_gdf = self.preprocess_dataset(dataset_gdf)

        dataset_path = self._get_dataset_path(filename, full_path=False)
        with LocalFile(dataset_path, mode="w", filesystem=self.storage.filesystem) as local_file:
            dataset_gdf.to_file(local_file.path, encoding="utf-8", driver="GPKG")

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
                self.vector_feature, path=path, layer=self.config.dataset_layer, config=self.sh_config
            )
            data_preparation_node = EONode(import_task, inputs=[create_node])
        else:
            input_task = LoadTask(
                self.storage.get_folder(self.config.input_folder_key, full_path=True),
                features=[self.vector_feature, FeatureType.BBOX],
                config=self.sh_config,
            )
            data_preparation_node = EONode(input_task)

        preprocess_node = self.get_prerasterization_node(data_preparation_node)

        rasterization_node = self.get_rasterization_node(preprocess_node)

        postprocess_node = self.get_postrasterization_node(rasterization_node)

        save_task = SaveTask(
            self.storage.get_folder(self.config.output_folder_key, full_path=True),
            features=self._get_output_features(),
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            config=self.sh_config,
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

        rasterization_nodes = [
            EONode(
                inputs=[previous_node],
                task=VectorToRasterTask(
                    vector_input=self.vector_feature,
                    raster_feature=column.output_feature,
                    values_column=column.values_column,
                    values=column.value,
                    buffer=column.polygon_buffer,
                    raster_resolution=column.resolution,
                    raster_dtype=np.dtype(column.dtype),
                    no_data_value=column.no_data_value,
                    overlap_value=column.overlap_value,
                ),
            )
            for column in self.config.columns
        ]

        if len(rasterization_nodes) == 1:
            return rasterization_nodes[0]
        return EONode(MergeEOPatchesTask(), inputs=rasterization_nodes)

    def get_postrasterization_node(self, previous_node: EONode) -> EONode:  # pylint: disable=no-self-use
        """Builds node with tasks to be applied after rasterization"""
        return previous_node

    @staticmethod
    def _parse_input_file(value: str) -> str:
        """Checks if given name ends with one of the supported file extensions"""
        if not value.lower().endswith((".geojson", ".shp", ".gpkg", ".gdb")):
            raise ValueError(f"Input file path {value} should be a GeoJSON, Shapefile, GeoPackage or GeoDataBase.")
        return value

    def _get_dataset_path(self, filename: str, full_path: bool = True) -> str:
        """Provides a path from where dataset should be loaded into the workflow"""
        if self.config.preprocess_dataset is not None:
            folder = self.storage.get_cache_folder(full_path=full_path)
            filename = f"preprocessed_{filename}"
            filename = (os.path.splitext(filename))[0] + ".gpkg"
        else:
            folder = self.storage.get_input_data_folder(full_path=full_path)

        if full_path:
            return join_path(folder, filename)
        return fs.path.combine(folder, filename)

    def _get_output_features(self) -> List[FeatureSpec]:
        """Lists all features that are to be saved upon the pipeline completion"""
        features: List[FeatureSpec] = [FeatureType.BBOX]
        features.extend(column.output_feature for column in self.config.columns)
        return features
