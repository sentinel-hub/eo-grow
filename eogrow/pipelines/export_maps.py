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

from eogrow.core.config import RawConfig

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
        no_data_value: Optional[float] = Field(description="No data value to be passed to GeoTIFFs")
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
            None, description="Which resampling to use in the cogification process for creating overviews."
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
            description=(
                "How many workers are used to parallelize merging of TIFFs. Uses all cores (of head node) by default."
                "Decreasing this should help with memory and disk-space issues."
            )
        )

    config: Schema
    MERGED_MAP_NAME = "merged"

    def __init__(self, config: Schema, raw_config: Optional[RawConfig] = None):
        super().__init__(config, raw_config)

        self.map_name = self.config.map_name or f"{self.config.feature[1]}.{MimeType.TIFF.extension}"

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Extracts and merges the data from EOPatches into a TIFF file.

        1. Extracts data from EOPatches via workflow into per-EOPatch tiffs.
        2. For each UTM zone:
            - Prepares tiffs for merging (transfers to local if needed).
            - Performs temporal split of tiffs if needed (assumption that all eopatches share the same timestamp)
            - Merges the tiffs
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

            exported_tiff_paths = [
                fs.path.join(folder, get_tiff_name(self.map_name, eopatch_name)) for eopatch_name in eopatch_list
            ]
            filesystem, geotiff_paths = self._prepare_files(exported_tiff_paths)

            if feature_type.is_timeless() or self.config.compress_temporally:
                geotiffs_per_time: Dict[Optional[dt.datetime], List[str]] = {None: geotiff_paths}
            else:
                geotiffs_per_time = self._split_patches_temporally(
                    crs, eopatch_list, output_folder, filesystem, geotiff_paths
                )

            LOGGER.info("Merging TIFF files.")
            output_paths = {
                time: get_tiff_name(self.map_name, self.MERGED_MAP_NAME, crs, time) for time in geotiffs_per_time
            }
            parallelize(
                self._combine_geotiffs,
                it.repeat(self.config),
                it.repeat(pickle_fs(filesystem)),
                geotiffs_per_time.values(),
                output_paths.values(),
                workers=self.config.merge_workers,
            )

            LOGGER.info("Finalizing output.")
            self._finalize_output(filesystem, output_paths, output_folder, exported_tiff_paths)

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
            # compress="LZW",
        )
        task_list.append(export_to_tiff_task)

        return EOWorkflow(linearly_connect_tasks(*task_list))

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        exec_args = super().get_execution_arguments(workflow)
        nodes = workflow.get_nodes()
        for node in nodes:
            if isinstance(node.task, ExportToTiffTask):
                for patch_name, single_exec_dict in zip(self.patch_list, exec_args):
                    single_exec_dict[node] = dict(filename=get_tiff_name(self.map_name, patch_name))

        return exec_args

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
        )
        return temp_fs, temp_geotiff_paths

    def _split_patches_temporally(
        self, crs: CRS, eopatch_list: List[str], output_folder: str, filesystem: FS, geotiff_paths: List[str]
    ) -> Dict[Optional[dt.datetime], List[str]]:
        geotiffs_per_time: Dict[Optional[dt.datetime], List[str]] = defaultdict(list)
        LOGGER.info("Splitting TIFF files temporally.")
        num_bands, timestamp = self._load_bands_and_timestamp(eopatch_list[0])  # assume eopatches share these
        filesystem.makedirs(output_folder, recreate=True)  # in case we use a temporary filesystem

        temporal_split_jobs = [
            self._prepare_split_jobs(filesystem, tiff_path, num_bands, timestamp, output_folder, crs)
            for tiff_path in geotiff_paths
        ]

        parallelize(self._execute_split_jobs, temporal_split_jobs, workers=None)

        for split_jobs in temporal_split_jobs:
            for job in split_jobs:
                geotiffs_per_time[job["time"]].append(job["output_path"])
        for path in geotiff_paths:
            filesystem.remove(path)
        return geotiffs_per_time

    def _load_bands_and_timestamp(self, eopatch_name: str) -> Tuple[int, List[dt.datetime]]:
        """Loads an eopatch to get information about number of bands and the timestamp."""
        path = fs.path.join(self.storage.get_folder(self.config.input_folder_key), eopatch_name)
        patch = EOPatch.load(path, (FeatureType.TIMESTAMP, self.config.feature), filesystem=self.storage.filesystem)
        if self.config.band_indices is not None:
            return len(self.config.band_indices), patch.timestamp
        return patch[self.config.feature].shape[-1], patch.timestamp

    def _prepare_split_jobs(
        self, filesystem: FS, tiff_path: str, num_bands: int, timestamp: List[dt.datetime], output_folder: str, crs: CRS
    ) -> List[dict]:
        """Prepares the specifics of the extraction, which requires difficult-to-pickle components."""
        tiff_name = fs.path.basename(tiff_path).replace(f".{MimeType.TIFF.extension}", "")

        jobs = []
        for i, time in enumerate(timestamp):
            extraction_path = fs.path.join(output_folder, get_tiff_name(self.map_name, tiff_name, crs, time))
            jobs.append(
                dict(
                    time=time,
                    bands=range(i * num_bands, (i + 1) * num_bands),
                    output_path=extraction_path,
                    sys_output_path=filesystem.getsyspath(extraction_path),
                    sys_input_path=filesystem.getsyspath(tiff_path),
                )
            )

        return jobs

    @staticmethod
    def _execute_split_jobs(jobs: List[dict]) -> None:
        """Executes all the jobs for a specific tiff. This prevents parallel processes fighting over IO to a tiff."""
        for job in jobs:
            extract_bands(job["sys_input_path"], job["sys_output_path"], job["bands"])

    @staticmethod
    def _combine_geotiffs(
        config: Schema, pickled_filesystem: bytes, tiff_paths: List[str], merged_map_path: str
    ) -> None:
        """Merges tiffs and cogifies them if needed. Also removes the pre-merge tiffs."""
        filesystem = unpickle_fs(pickled_filesystem)
        merge_tiffs(
            map(filesystem.getsyspath, tiff_paths),
            filesystem.getsyspath(merged_map_path),
            nodata=config.no_data_value,
            dtype=config.map_dtype,
            warp_resampling=config.warp_resampling,
        )
        for tiff_path in tiff_paths:
            filesystem.remove(tiff_path)

        if config.cogify:
            cogify_inplace(
                filesystem.getsyspath(merged_map_path),
                blocksize=1024,
                nodata=config.no_data_value,
                dtype=config.map_dtype,
                resampling=config.cogification_resampling,
            )

    def _finalize_output(
        self,
        filesystem: FS,
        merged_maps_paths: Dict[Optional[dt.datetime], str],
        output_folder: str,
        exported_tiff_paths: List[str],
    ) -> None:
        """Renames (or transfers in case of temporal FS) the files to the expected output files.

        If a temporal filesystem was used, the exported tiffs are cleaned from the storage and the filesystem closed.
        """
        for timestamp, map_path in tqdm(merged_maps_paths.items()):
            if timestamp is None:
                output_path = fs.path.join(output_folder, self.map_name)
            else:
                time_output_folder = fs.path.join(output_folder, timestamp.strftime(TIMESTAMP_FORMAT))
                self.storage.filesystem.makedirs(time_output_folder, recreate=True)
                output_path = fs.path.join(time_output_folder, self.map_name)

            fs.copy.copy_file(filesystem, map_path, self.storage.filesystem, output_path)
            filesystem.remove(map_path)

        LOGGER.info("Merged maps are saved in %s", get_full_path(self.storage.filesystem, output_folder))

        if isinstance(filesystem, TempFS):
            filesystem.close()

            # for non-tempfs runs this is already done right after merging
            LOGGER.info("Remove per-eopatch tiffs from storage.")
            parallelize(self.storage.filesystem.remove, exported_tiff_paths, workers=None, multiprocess=False)


def get_tiff_name(map_name: str, name: str, crs: Optional[CRS] = None, time: Optional[dt.datetime] = None) -> str:
    """Creates a name of a geotiff image"""
    map_name = map_name.replace(f".{MimeType.TIFF.extension}", "")
    base = f"{map_name}_{name}"
    if crs is not None:
        base += f"_UTM_{crs.epsg}"
    if time is not None:
        base += f"_{time.strftime(TIMESTAMP_FORMAT)}"
    return f"{base}.{MimeType.TIFF.extension}"
