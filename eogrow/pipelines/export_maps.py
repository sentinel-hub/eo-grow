"""
Module implementing export_maps pipelines
"""
import datetime as dt
import itertools as it
import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import fs
import fs.copy
import numpy as np
import rasterio
from fs.base import FS
from fs.tempfs import TempFS
from pydantic import Field
from tqdm.auto import tqdm

from eolearn.core import EONode, EOPatch, EOTask, EOWorkflow, FeatureType, LoadTask, linearly_connect_tasks
from eolearn.core.utils.fs import get_full_path, pickle_fs, unpickle_fs
from eolearn.core.utils.parallelize import parallelize
from eolearn.features import LinearFunctionTask
from eolearn.io import ExportToTiffTask
from sentinelhub import CRS, MimeType

from ..core.pipeline import Pipeline
from ..utils.map import CogifyResamplingOptions, WarpResamplingOptions, cogify_inplace, extract_bands, merge_tiffs
from ..utils.types import Feature

LOGGER = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


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
        map_name: Optional[str] = Field(regex=r".+\." + MimeType.TIFF.extension + r"?\b")  # noqa
        map_dtype: Literal["int8", "int16", "uint8", "uint16", "float32"]
        no_data_value: int = Field(0, description="No data value to be passed to GeoTIFFs")
        scale_factor: Optional[float] = Field(description="Feature will be multiplied by this value at export")
        band_indices: Optional[List[int]] = Field(
            description="A list of band indices to be exported for the export feature. Default is all bands"
        )
        warp_resampling: WarpResamplingOptions = Field(
            None, description="The resampling method used when warping, useful for pixel misalignment"
        )

        cogify: bool = Field(
            False, description="Whether exported GeoTIFFs will be converted into Cloud Optimized GeoTIFFs (COG)"
        )
        cogification_resampling: CogifyResamplingOptions = Field(
            None, description="Which resampling to use in the cogification process."
        )

        force_local_copies: bool = Field(
            False,
            description=(
                "By default copying to local temporary folder will happen only if an AWS S3 path is configured."
                " With this parameter you force to always make copies."
            ),
        )
        compress_temporally: bool = Field(
            False,
            description=(
                "Temporal features are by default exported as multiple TIFF files in a per-timestamp manner. Enabling "
                "this parameter results in a single TIFF with the same band order as the one of `ExportToTiffTask`."
            ),
        )
        skip_existing: Literal[False] = False
        merge_workers: Optional[int] = Field(
            description="How many workers to use to parallelize (over the temporal dimension) merging."
        )

    config: Schema
    MERGED_MAP_NAME = "merged"

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Extracts and merges the data from EOPatches into a TIFF file.

        1. Extracts data from EOPatches via workflow into per-EOPatch tiffs.
        2. For each UTM zone:
            - Prepares tiffs for merging (transfers to local if needed).
            - Merges the tiffs
            - Performs temporal split of tiffs if needed (assumption that all eopatches share the same timestamp)
            - Cogification is done if requested.
            - The output files are finalized (renamed/transferred) and per-EOPatch tiffs are cleaned.

        """

        successful, failed = super().run_procedure()

        if not successful:
            raise ValueError("Failed to extract tiff files from any of EOPatches.")

        feature_type, _ = self.config.feature
        folder = self.storage.get_folder(self.config.output_folder_key)
        crs_eopatch_dict = self.eopatch_manager.split_by_utm(successful)

        for crs, eopatch_list in crs_eopatch_dict.items():
            LOGGER.info("Processing for UTM %d", crs.epsg)

            output_folder = fs.path.join(folder, f"UTM_{crs.epsg}")
            # manually make subfolder, otherwise things fail on S3 in later steps
            self.storage.filesystem.makedirs(output_folder, recreate=True)

            exported_tiff_paths = [fs.path.join(folder, self.get_geotiff_name(name)) for name in eopatch_list]

            filesystem, geotiff_paths = self._prepare_files(exported_tiff_paths)

            geotiffs_per_time: Dict[Optional[dt.datetime], List[str]] = defaultdict(list)
            if feature_type.is_timeless() or self.config.compress_temporally:
                geotiffs_per_time[None] = geotiff_paths
            else:
                LOGGER.info("Splitting TIFF files temporally.")
                timestamp = self._load_timestamp(eopatch_list[0])  # we assume all eopatches share the same timestamp
                split_tiffs = [
                    self._split_temporally(filesystem, tiff_path, timestamp, output_folder, crs)
                    for tiff_path in geotiff_paths
                ]

                for split in split_tiffs:
                    for path, time in split:
                        geotiffs_per_time[time].append(path)

            LOGGER.info("Merging TIFF files.")
            output_paths = {time: self.get_geotiff_name(self.MERGED_MAP_NAME, crs, time) for time in geotiffs_per_time}
            parallelize(
                self._combine_geotiffs,
                it.repeat(self.config),
                it.repeat(pickle_fs(filesystem)),
                geotiffs_per_time.values(),
                output_paths.values(),
                workers=self.config.merge_workers,
            )

            LOGGER.info("Finalizing output files.")
            self._finalize_output_files(filesystem, output_paths, output_folder)
            if isinstance(filesystem, TempFS):
                filesystem.close()

                # for non-tempfs runs this is already done right after merging
                LOGGER.info("Remove per-eopatch tiffs for UTM %d from storage.", crs.epsg)
                parallelize(
                    self.storage.filesystem.remove,
                    exported_tiff_paths,
                    workers=None,
                    multiprocess=False,
                    desc="Remove tiffs",
                )

        return successful, failed

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
            folder=self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            no_data_value=self.config.no_data_value,
            image_dtype=np.dtype(self.config.map_dtype),
            band_indices=self.config.band_indices,
            compress="LZW",
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

    def get_geotiff_name(self, name: str, crs: Optional[CRS] = None, time: Optional[dt.datetime] = None) -> str:
        """Creates a name of a geotiff image"""
        base = f"{self._get_map_name(with_extension=False)}_{name}"
        if crs is not None:
            base += f"_UTM_{crs.epsg}"
        if time is not None:
            base += f"_{time.strftime(TIMESTAMP_FORMAT)}"
        return f"{base}.{MimeType.TIFF.extension}"

    def _get_map_name(self, with_extension: bool = True) -> str:
        name = self.config.map_name or f"{self.config.feature[1]}.{MimeType.TIFF.extension}"
        return name if with_extension else name.replace(f".{MimeType.TIFF.extension}", "")

    def _prepare_files(self, geotiff_paths: List[str]) -> Tuple[FS, List[str]]:
        """Returns system paths of geotiffs and output file that can be used to merge maps.

        If required files are copied locally and a temporary filesystem object is returned.
        """
        make_local_copies = self.storage.is_on_aws() or self.config.force_local_copies

        if not make_local_copies:
            return self.storage.filesystem, geotiff_paths

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
        return temp_fs, temp_geotiff_paths

    def _load_timestamp(self, eopatch_name: str) -> List[dt.datetime]:
        path = fs.path.join(self.storage.get_folder(self.config.input_folder_key), eopatch_name)
        patch = EOPatch.load(path, FeatureType.TIMESTAMP, filesystem=self.storage.filesystem)
        return patch.timestamp

    def _split_temporally(
        self, filesystem: FS, tiff_path: str, timestamp: List[dt.datetime], output_folder: str, crs: CRS
    ) -> List[Tuple[str, dt.datetime]]:
        """Splits the merged tiff into multiple tiffs, one per timestamp."""
        # TODO: output_folder no longer needed here
        filesystem.makedirs(output_folder, recreate=True)  # in case we use a temporary filesystem
        tiff_name = fs.path.basename(tiff_path).replace(f".{MimeType.TIFF.extension}", "")

        with filesystem.openbin(tiff_path) as file_handle:
            with rasterio.open(file_handle) as map_src:
                num_bands = map_src.count // len(timestamp)

        outputs: List[Tuple[str, dt.datetime]] = []
        for i, time in enumerate(timestamp):
            name = self.get_geotiff_name(tiff_name, crs, time)
            extraction_path = fs.path.join(output_folder, name)
            bands = range(i * num_bands, (i + 1) * num_bands)
            extract_bands(filesystem.getsyspath(tiff_path), filesystem.getsyspath(extraction_path), bands, quiet=True)
            outputs.append((extraction_path, time))

        filesystem.remove(tiff_path)
        return outputs

    @staticmethod
    def _combine_geotiffs(
        config: Schema, pickled_filesystem: bytes, tiff_paths: List[str], merged_map_path: str
    ) -> None:
        """Merges tiffs and cogifies them if needed. Also removes the pre-merge tiffs."""
        filesystem = unpickle_fs(pickled_filesystem)
        merge_tiffs(
            list(map(filesystem.getsyspath, tiff_paths)),
            filesystem.getsyspath(merged_map_path),
            overwrite=True,
            nodata=config.no_data_value,
            dtype=config.map_dtype,
            warp_resampling=config.warp_resampling,
            quiet=True,
        )
        for tiff_path in tiff_paths:
            filesystem.remove(tiff_path)

        if config.cogify:
            cogify_inplace(
                filesystem.getsyspath(merged_map_path),
                blocksize=1024,
                nodata=config.no_data_value,
                dtype=config.map_dtype,
                quiet=True,
                resampling=config.cogification_resampling,
            )

    def _finalize_output_files(
        self, filesystem: FS, output_paths: Dict[Optional[dt.datetime], str], output_folder: str
    ) -> None:
        """Renames (or transfers in case of temporal FS) the files to the expected output files."""
        for timestamp, map_path in tqdm(output_paths.items(), desc="Finalizing output files"):
            if timestamp is None:
                output_map_path = fs.path.join(output_folder, self._get_map_name())
            else:
                timestamp_output_folder = fs.path.join(output_folder, timestamp.strftime(TIMESTAMP_FORMAT))
                self.storage.filesystem.makedirs(timestamp_output_folder, recreate=True)
                output_map_path = fs.path.join(timestamp_output_folder, self._get_map_name())

            fs.copy.copy_file(filesystem, map_path, self.storage.filesystem, output_map_path)
            filesystem.remove(map_path)

        LOGGER.info("Merged maps are saved in %s", get_full_path(self.storage.filesystem, output_folder))
