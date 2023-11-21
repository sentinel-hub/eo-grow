"""Implements a pipeline for exporting data to TIFF files, can be used to prepare BYOC tiles."""

from __future__ import annotations

import datetime as dt
import itertools as it
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Iterable, List, Literal, Optional

import fs
import fs.copy
import numpy as np
from fs.base import FS
from fs.tempfs import TempFS
from pydantic import Field
from tqdm.auto import tqdm

from eolearn.core import EOPatch, EOTask, EOWorkflow, LoadTask, linearly_connect_tasks
from eolearn.core.types import Feature
from eolearn.core.utils.fs import get_full_path, pickle_fs, unpickle_fs
from eolearn.core.utils.parallelize import parallelize
from eolearn.io import ExportToTiffTask
from sentinelhub import CRS, MimeType

from eogrow.core.config import RawConfig

from ..core.pipeline import Pipeline
from ..tasks.common import LinearFunctionTask
from ..types import ExecKwargs, PatchList
from ..utils.eopatch_list import group_by_crs
from ..utils.map import CogifyResamplingOptions, WarpResamplingOptions, cogify_inplace, extract_bands, merge_tiffs
from ..utils.validators import ensure_storage_key_presence

LOGGER = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


class ExportMapsPipeline(Pipeline):
    """Pipeline to export a feature into a tiff map"""

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="The storage manager key pointing to the input folder for the export maps pipeline."
        )
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        output_folder_key: str = Field(
            description=(
                "The storage manager key pointing to the output folder for the maps in the export maps pipeline."
            )
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        feature: Feature
        map_name: Optional[str] = Field(regex=r".+\." + MimeType.TIFF.extension + r"?\b")
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
        interim_results_suffix: str = Field(
            "",
            description=(
                "Has no effect on end results. Adds a suffix to names of temporary files in order to avoid clashes"
                " from pipelines working in parallel on same maps (e.g. exporting same map for different timestamps)."
            ),
        )
        split_per_timestamp: bool = Field(
            True,
            description=(
                "Temporal features are by default exported as multiple TIFF files in a per-timestamp manner. Disabling "
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

    def __init__(self, config: Schema, raw_config: RawConfig | None = None):
        super().__init__(config, raw_config)

        self.map_name = self.config.map_name or f"{self.config.feature[1]}.{MimeType.TIFF.extension}"
        self.get_tiff_name = partial(get_tiff_name, self.map_name, suffix=self.config.interim_results_suffix)

    def run_procedure(self) -> tuple[list[str], list[str]]:
        """Extracts and merges the data from EOPatches into a TIFF file.

        1. Extracts data from EOPatches via workflow into per-EOPatch tiffs.
        2. For each UTM zone:
            - Prepares tiffs for merging (transfers to local if needed).
            - Performs temporal split of tiffs if needed (assumption that all eopatches share the same timestamps)
            - Merges the tiffs
            - Cogification is done if requested.
            - The output files are finalized (renamed/transferred) and per-EOPatch tiffs are cleaned.

        """

        workflow = self.build_workflow()
        patch_list = self.get_patch_list()
        exec_args = self.get_execution_arguments(workflow, patch_list)

        successful, failed, _ = self.run_execution(workflow, exec_args)

        if not successful:
            raise RuntimeError("Failed to extract tiff files from any of EOPatches.")

        feature_type, _ = self.config.feature
        output_folder = self.storage.get_folder(self.config.output_folder_key)

        for crs, eopatch_name_list in group_by_crs(patch_list).items():
            LOGGER.info("Processing CRS %d", crs.epsg)

            exported_tiff_paths = [
                fs.path.join(output_folder, self.get_tiff_name(patch_name)) for patch_name in eopatch_name_list
            ]
            filesystem, geotiff_paths = self._prepare_files(exported_tiff_paths)

            crs_output_folder = fs.path.join(output_folder, f"UTM_{crs.epsg}")
            filesystem.makedirs(crs_output_folder, recreate=True)

            if feature_type.is_timeless() or not self.config.split_per_timestamp:
                merged_map_name = self.get_tiff_name(self.MERGED_MAP_NAME, crs)
                combine_tiffs_jobs = [CombineTiffsJob(geotiff_paths, merged_map_name, time=None)]
            else:
                time_to_tiffs_map = self._split_patches_temporally(
                    filesystem, crs_output_folder, geotiff_paths, some_eopatch=eopatch_name_list[0], crs=crs
                )
                map_name_maker = partial(self.get_tiff_name, self.MERGED_MAP_NAME, crs)

                combine_tiffs_jobs = [
                    CombineTiffsJob(tiffs, map_name_maker(time), time) for time, tiffs in time_to_tiffs_map.items()
                ]

            LOGGER.info("Merging TIFF files.")
            parallelize(
                partial(self._combine_geotiffs, self.config, pickle_fs(filesystem)),
                combine_tiffs_jobs,
                workers=self.config.merge_workers,
            )

            LOGGER.info("Finalizing output.")
            merged_maps = {job.time: job.output_path for job in combine_tiffs_jobs}
            self._finalize_output(filesystem, merged_maps, crs_output_folder, exported_tiff_paths)

        return successful, failed

    def build_workflow(self) -> EOWorkflow:
        load_task = LoadTask(
            self.storage.get_folder(self.config.input_folder_key),
            filesystem=self.storage.filesystem,
            features=[self.config.feature],
        )
        task_list: list[EOTask] = [load_task]

        if self.config.scale_factor is not None:
            rescale_task = LinearFunctionTask(self.config.feature, slope=self.config.scale_factor)
            task_list.append(rescale_task)

        export_to_tiff_task = ExportToTiffTask(
            self.config.feature,
            self.storage.get_folder(self.config.output_folder_key),
            filesystem=self.storage.filesystem,
            no_data_value=self.config.no_data_value,
            image_dtype=np.dtype(self.config.map_dtype),
            band_indices=self.config.band_indices,
        )
        task_list.append(export_to_tiff_task)

        return EOWorkflow(linearly_connect_tasks(*task_list))

    def get_execution_arguments(self, workflow: EOWorkflow, patch_list: PatchList) -> ExecKwargs:
        exec_args = super().get_execution_arguments(workflow, patch_list)
        nodes = workflow.get_nodes()
        for node in nodes:
            if isinstance(node.task, ExportToTiffTask):
                for patch_name, patch_args in exec_args.items():
                    patch_args[node] = dict(filename=self.get_tiff_name(patch_name))

        return exec_args

    def _prepare_files(self, geotiff_paths: list[str]) -> tuple[FS, list[str]]:
        """Returns a filesystem and appropriate paths to the geotiffs to use in the pipeline.

        If required files are copied locally and a temporary filesystem object is returned.
        """
        make_local_copies = self.storage.is_on_s3() or self.config.force_local_copies

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
        self, filesystem: FS, output_folder: str, geotiff_paths: list[str], some_eopatch: str, crs: CRS
    ) -> dict[dt.datetime | None, list[str]]:
        """Splits multi-temporal tiffs into separate tiffs according to the temporal dimensions.

        The jobs for splitting tiffs are grouped together per input file. This ensures that two processes don't try
        to use GDAL on the same input file (avoiding IO blocks).

        Because the rest of the pipeline groups parallelization jobs per time, the function returns `time_to_tiffs_map`
        that groups output tiffs of the same timestamp.
        """
        LOGGER.info("Splitting TIFF files temporally.")
        num_bands, timestamps = self._extract_num_bands_and_timestamps(some_eopatch)  # assume eopatches share these

        def get_extraction_path(input_path: str, time: dt.datetime) -> str:
            """Ensures that after extraction the files have a time-suffix."""
            tiff_name = _strip_tiff_extension(fs.path.basename(input_path))
            return fs.path.join(output_folder, self.get_tiff_name(tiff_name, crs, time))

        time_to_tiffs_map: dict[dt.datetime | None, list[str]] = defaultdict(list)
        grouped_split_jobs = []  # these are grouped per input file to avoid IO blocking of processes
        for tiff_path in geotiff_paths:
            tiff_sys_path = filesystem.getsyspath(tiff_path)

            jobs = []
            for i, time in enumerate(timestamps):
                extraction_path = get_extraction_path(tiff_path, time)
                bands = range(i * num_bands, (i + 1) * num_bands)
                jobs.append(SplitTiffsJob(tiff_sys_path, bands, filesystem.getsyspath(extraction_path)))

                time_to_tiffs_map[time].append(extraction_path)

            grouped_split_jobs.append(jobs)

        parallelize(self._execute_split_jobs, grouped_split_jobs, workers=None)
        parallelize(filesystem.remove, geotiff_paths, workers=None, multiprocess=False)

        return time_to_tiffs_map

    def _extract_num_bands_and_timestamps(self, eopatch_name: str) -> tuple[int, list[dt.datetime]]:
        """Loads an eopatch to get information about number of bands and the timestamps."""
        path = fs.path.join(self.storage.get_folder(self.config.input_folder_key), eopatch_name)
        patch = EOPatch.load(path, [self.config.feature], filesystem=self.storage.filesystem)
        if self.config.band_indices is not None:
            return len(self.config.band_indices), patch.get_timestamps()
        return patch[self.config.feature].shape[-1], patch.get_timestamps()

    @staticmethod
    def _execute_split_jobs(jobs: list[SplitTiffsJob]) -> None:
        """Executes all the jobs for a specific tiff. This prevents parallel processes fighting over IO to a tiff."""
        for job in jobs:
            extract_bands(job.input_path, job.output_path, job.bands)

    @staticmethod
    def _combine_geotiffs(config: Schema, pickled_filesystem: bytes, job: CombineTiffsJob) -> None:
        """Merges tiffs and cogifies them if needed. Also removes the pre-merge tiffs."""
        filesystem = unpickle_fs(pickled_filesystem)
        merge_tiffs(
            map(filesystem.getsyspath, job.input_paths),
            filesystem.getsyspath(job.output_path),
            nodata=config.no_data_value,
            dtype=config.map_dtype,
            warp_resampling=config.warp_resampling,
        )
        for tiff_path in job.input_paths:
            filesystem.remove(tiff_path)

        if config.cogify:
            cogify_inplace(
                filesystem.getsyspath(job.output_path),
                blocksize=1024,
                nodata=config.no_data_value,
                dtype=config.map_dtype,
                resampling=config.cogification_resampling,
            )

    def _finalize_output(
        self,
        filesystem: FS,
        merged_maps_paths: dict[dt.datetime | None, str],
        output_folder: str,
        exported_tiff_paths: list[str],
    ) -> None:
        """Renames (or transfers in case of temporal FS) the files to the expected output files.

        If a temporal filesystem was used, the exported tiffs are cleaned from the storage and the filesystem closed.
        """
        self.storage.filesystem.makedirs(output_folder, recreate=True)
        for timestamp, map_path in tqdm(merged_maps_paths.items()):
            if timestamp is None:
                output_path = fs.path.join(output_folder, self.map_name)
            else:
                time_output_folder = fs.path.join(output_folder, timestamp.strftime(TIMESTAMP_FORMAT))
                self.storage.filesystem.makedirs(time_output_folder, recreate=True)
                output_path = fs.path.join(time_output_folder, self.map_name)

            fs.copy.copy_file(filesystem, map_path, self.storage.filesystem, output_path)
            filesystem.remove(map_path)

        LOGGER.info("Merged maps saved to %s", get_full_path(self.storage.filesystem, output_path))

        if isinstance(filesystem, TempFS):
            filesystem.close()

            # for non-tempfs runs this is already done right after merging
            LOGGER.info("Remove per-eopatch tiffs from storage.")
            parallelize(self.storage.filesystem.remove, exported_tiff_paths, workers=None, multiprocess=False)


@dataclass
class SplitTiffsJob:
    """Describes which bands of input path to extract to the output path."""

    input_path: str
    bands: Iterable[int]
    output_path: str


@dataclass
class CombineTiffsJob:
    """Describes which tiffs to merge, what the output path is, and the time the merged tiff represents.

    The time is relevant in order to correctly place the finalized tiff. If left out then the tiffs are either
    timeless or not split temporally.
    """

    input_paths: Iterable[str]
    output_path: str
    time: dt.datetime | None


def _strip_tiff_extension(path: str) -> str:
    return path.replace(f".{MimeType.TIFF.extension}", "")


def get_tiff_name(
    map_name: str, name: str, crs: CRS | None = None, time: dt.datetime | None = None, suffix: str = ""
) -> str:
    """Creates a name of a geotiff image"""
    base = f"{_strip_tiff_extension(map_name)}_{name}"
    if crs is not None:
        base += f"_UTM_{crs.epsg}"
    if time is not None:
        base += f"_{time.strftime(TIMESTAMP_FORMAT)}"
    if suffix:
        base += f"_{suffix}"
    return f"{base}.{MimeType.TIFF.extension}"
