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
from ..utils.map import cogify_inplace, merge_tiffs
from ..utils.types import Feature

LOGGER = logging.getLogger(__name__)


class ExportMapsPipeline(Pipeline):
    """Pipeline to export a feature into a tiff map"""

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
        """Extracts and merges the data from EOPatches into a TIFF file.

        1. Extracts data from EOPatches via workflow into per-EOPatch tiffs.
        2. For each UTM zone:
            - Prepares tiffs for merging (transfers to local if needed).
            - Merges the tiffs, cogification is done if requested.
            - The output file is finalized (transferred if needed) and per-EOPatch tiffs are cleaned.

        """

        successful, failed = super().run_procedure()

        if not successful:
            raise ValueError("Failed to extract tiff files from any of EOPatches.")

        folder = self._get_output_folder()
        crs_eopatch_dict = self.eopatch_manager.split_by_utm(successful)

        # TODO: This could be parallelized per-crs
        for crs, eopatch_list in crs_eopatch_dict.items():
            subfolder = f"UTM_{crs.epsg}"
            # manually make subfolder, otherwise things fail on S3 in later steps
            self.storage.filesystem.makedirs(fs.path.join(folder, subfolder), recreate=True)

            map_name = self.config.map_name or f"{self.config.feature[1]}.tiff"
            merged_map_path = fs.path.join(folder, subfolder, map_name)
            geotiff_paths = [fs.path.join(folder, self.get_geotiff_name(name)) for name in eopatch_list]

            temp_fs, geotiff_sys_paths, map_sys_path = self._prepare_input_files(geotiff_paths, merged_map_path)

            merge_tiffs(
                geotiff_sys_paths,
                map_sys_path,
                overwrite=True,
                delete_input=False,
                nodata=self.config.no_data_value,
                dtype=self.config.map_dtype,
            )

            if self.config.cogify:
                cogify_inplace(
                    map_sys_path, blocksize=1024, nodata=self.config.no_data_value, dtype=self.config.map_dtype
                )

            self._finalize_output_files(temp_fs, map_sys_path, merged_map_path)

            parallelize(
                self.storage.filesystem.remove,
                geotiff_paths,
                workers=None,
                multiprocess=False,
                desc=f"Remove per-eopatch tiffs for UTM {crs.epsg}",
            )

        return successful, failed

    def _finalize_output_files(self, temp_fs: Optional[TempFS], map_sys_path: str, merged_map_path: str) -> None:
        """In case a temporary filesystem was used the output files need to be transferred to the correct location."""
        if temp_fs is not None:
            temp_map_path = fs.path.frombase(temp_fs.getsyspath(""), map_sys_path)
            fs.copy.copy_file(temp_fs, temp_map_path, self.storage.filesystem, merged_map_path)
            temp_fs.close()
        LOGGER.info("Merged map is saved at %s", get_full_path(self.storage.filesystem, merged_map_path))

    def build_workflow(self) -> EOWorkflow:
        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key),
            filesystem=self.storage.filesystem,
            features=[self.config.feature, FeatureType.BBOX],
        )
        task_list: List[EOTask] = [load_task]

        if self.config.scale_factor is not None:
            rescale_task = LinearFunctionTask(self.config.feature, slope=self.config.scale_factor)
            task_list.append(rescale_task)

        export_to_tiff_task = ExportToTiffTask(
            self.config.feature,
            folder=self._get_output_folder(),
            filesystem=self.storage.filesystem,
            no_data_value=self.config.no_data_value,
            image_dtype=np.dtype(self.config.map_dtype),
            band_indices=self.config.band_indices,
        )
        task_list.append(export_to_tiff_task)

        return EOWorkflow(linearly_connect_tasks(*task_list))

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        exec_args = super().get_execution_arguments(workflow)
        nodes = workflow.get_nodes()
        for node in nodes:
            if isinstance(node.task, ExportToTiffTask):
                for name, single_exec_dict in zip(self.patch_list, exec_args):
                    single_exec_dict[node] = dict(filename=self.get_geotiff_name(name))

        return exec_args

    def get_geotiff_name(self, name: str) -> str:
        """Creates a unique name of a geotiff image"""
        return f"{name}_{self.__class__.__name__}_{self.pipeline_id}.tiff"

    def _get_output_folder(self) -> str:
        """Designates to place the tiffs into a subfolder of output_folder_key named after the feature."""
        _, feature_name = self.config.feature
        return fs.path.join(self.storage.get_folder(self.config.output_folder_key), feature_name)

    def _prepare_files(self, geotiff_paths: List[str], output_file: str) -> Tuple[Optional[TempFS], List[str], str]:
        """Returns system paths of geotiffs and output file that can be used to merge maps.

        If required files are copied locally and a temporary filesystem object is returned.
        """
        make_local_copies = self.storage.is_on_aws() or self.config.force_local_copies

        if make_local_copies:
            temp_fs = TempFS(identifier="_merge_maps_temp")
            temp_geotiff_paths = [fs.path.basename(path) for path in geotiff_paths]

            LOGGER.info("Copying tiffs to a temporary local folder %s", temp_fs.getsyspath("/"))
            parallelize(
                fs.copy.copy_file,
                it.repeat(self.storage.filesystem),
                geotiff_paths,
                it.repeat(temp_fs),
                temp_geotiff_paths,
                workers=None,
                multiprocess=False,
                desc="Making local copies",
            )

            sys_paths = [temp_fs.getsyspath(path) for path in temp_geotiff_paths]

            temp_map_path = fs.path.basename(output_file)
            sys_map_path = temp_fs.getsyspath(temp_map_path)
            return temp_fs, sys_paths, sys_map_path

        sys_paths = [self.storage.filesystem.getsyspath(path) for path in geotiff_paths]
        sys_map_path = self.storage.filesystem.getsyspath(output_file)
        return None, sys_paths, sys_map_path
