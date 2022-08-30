"""
Module implementing export_maps pipelines
"""
import itertools as it
import logging
from typing import Dict, List, Literal, Optional, Tuple

import fs
import fs.copy
import numpy as np
from fs.tempfs import TempFS
from pydantic import Field

from eolearn.core import EONode, EOTask, EOWorkflow, FeatureType, LoadTask, linearly_connect_tasks
from eolearn.core.utils.fs import get_full_path
from eolearn.core.utils.parallelize import parallelize
from eolearn.features import LinearFunctionTask
from eolearn.io import ExportToTiffTask

from ..core.pipeline import Pipeline
from ..utils.map import merge_maps
from ..utils.types import Feature

LOGGER = logging.getLogger(__name__)


class ExportMapsPipeline(Pipeline):
    """Pipeline to export a feature into a tiff map"""

    _merge_map_function = staticmethod(merge_maps)

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder for the export maps pipeline."
        )
        output_folder_key: str = Field(
            description=(
                "The storage manager key pointing to the output folder for the maps in the export maps pipeline."
            )
        )

        feature: Feature
        map_name: Optional[str] = Field(regex=r".+\.tiff?\b")  # noqa
        map_dtype: Literal["int8", "int16", "uint8", "uint16", "float32"]
        no_data_value: int = Field(0, description="No data value to be passed to GeoTIFFs")
        scale_factor: Optional[float] = Field(description="Feature will be multiplied by this value at export")
        band_indices: Optional[List[int]] = Field(
            description="A list of band indices to be exported for the export feature. Default is all bands"
        )

        cogify: bool = Field(
            False, description="Whether exported GeoTIFFs will be converted into Cloud Optimized GeoTIFFs (COG)"
        )
        force_local_copies: bool = Field(
            False,
            description=(
                "By default copying to local temporary folder will happen only if an AWS S3 path is configured."
                " With this parameter you force to always make copies."
            ),
        )
        skip_existing: Literal[False] = False

    config: Schema

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Procedure which downloads satellite data"""
        successful, failed = super().run_procedure()

        _, feature_name = self.config.feature
        self.create_maps(feature_name, successful)

        return successful, failed

    def build_workflow(self) -> EOWorkflow:
        """Method where workflow is constructed"""
        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key),
            filesystem=self.storage.filesystem,
            features=[self.config.feature, FeatureType.BBOX],
        )
        task_list: List[EOTask] = [load_task]

        if self.config.scale_factor is not None:
            rescale_task = LinearFunctionTask(self.config.feature, slope=self.config.scale_factor)
            task_list.append(rescale_task)

        _, feature_name = self.config.feature
        folder = fs.path.join(self.storage.get_folder(self.config.output_folder_key), feature_name)
        export_to_tiff_task = ExportToTiffTask(
            self.config.feature,
            folder=folder,
            filesystem=self.storage.filesystem,
            no_data_value=self.config.no_data_value,
            image_dtype=np.dtype(self.config.map_dtype),
            band_indices=self.config.band_indices,
        )
        task_list.append(export_to_tiff_task)

        return EOWorkflow(linearly_connect_tasks(*task_list))

    def create_maps(self, feature_name: str, patch_names: List[str]) -> None:
        """A method which creates a joined GeoTIFF maps from a list of exported GeoTIFFs"""
        if not patch_names:
            raise ValueError("Cannot create map with an empty list of EOPatch names")

        folder = fs.path.join(self.storage.get_folder(self.config.output_folder_key), feature_name)

        crs_eopatch_dict = self.eopatch_manager.split_by_utm(patch_names)

        # TODO: This could be parallelized per-crs
        for crs, eopatch_list in crs_eopatch_dict.items():
            subfolder = f"UTM_{crs.epsg}"
            map_name = self.config.map_name or f"{feature_name}.tiff"

            # manually make subfolder, otherwise things fail on S3 in later steps
            self.storage.filesystem.makedirs(fs.path.join(folder, subfolder), recreate=True)
            self._create_single_map(folder, eopatch_list, subfolder, map_name)

    def _create_single_map(self, folder: str, eopatch_list: List[str], subfolder: str, map_name: str) -> None:
        """Creates a single map"""
        make_local_copies = self.storage.is_on_aws() or self.config.force_local_copies

        geotiff_names = [self.get_geotiff_name(name) for name in eopatch_list]
        geotiff_paths = [fs.path.join(folder, name) for name in geotiff_names]
        merged_map_path = fs.path.join(folder, subfolder, map_name)

        if make_local_copies:
            temp_fs = TempFS(identifier="_merge_maps_temp")

            LOGGER.info("Copying tiffs to a temporary local folder %s", temp_fs.getsyspath("/"))
            parallelize(
                fs.copy.copy_file,
                it.repeat(self.storage.filesystem),
                geotiff_paths,
                it.repeat(temp_fs),
                geotiff_names,
                workers=None,
                multiprocess=False,
                desc="Local copies",
            )

            sys_paths = [temp_fs.getsyspath(name) for name in geotiff_names]
            sys_map_path = temp_fs.getsyspath(map_name)
        else:
            sys_paths = [self.storage.filesystem.getsyspath(path) for path in geotiff_paths]
            sys_map_path = self.storage.filesystem.getsyspath(merged_map_path)

        self._merge_map_function(
            sys_paths,
            sys_map_path,
            nodata=self.config.no_data_value,
            dtype=self.config.map_dtype,
            cogify=self.config.cogify,
            delete_input=False,
        )

        if make_local_copies:
            fs.copy.copy_file(temp_fs, map_name, self.storage.filesystem, merged_map_path)
            temp_fs.close()
        LOGGER.info("Merged map is saved at %s", get_full_path(self.storage.filesystem, merged_map_path))

        parallelize(
            self.storage.filesystem.remove, geotiff_paths, workers=None, multiprocess=False, desc="Deleting temp tiffs"
        )
        LOGGER.info("Cleaned original temporary maps")

    def get_geotiff_name(self, name: str) -> str:
        """Creates a unique name of a geotiff image"""
        return f"{name}_{self.__class__.__name__}_{self.pipeline_id}.tiff"

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        exec_args = super().get_execution_arguments(workflow)
        nodes = workflow.get_nodes()
        for node in nodes:
            if isinstance(node.task, ExportToTiffTask):
                for name, single_exec_dict in zip(self.patch_list, exec_args):
                    single_exec_dict[node] = dict(filename=self.get_geotiff_name(name))

        return exec_args
