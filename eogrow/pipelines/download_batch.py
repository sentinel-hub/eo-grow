"""Download pipeline that works with Sentinel Hub batch service."""

from __future__ import annotations

import logging
import sqlite3
import time
from functools import wraps
from typing import Any, Callable, List, Literal, Optional, TypeVar

import fs
import pandas as pd
import requests
from geopandas import GeoDataFrame
from pydantic import Field, validator
from typing_extensions import ParamSpec

from sentinelhub import (
    CRS,
    BatchProcessClient,
    BatchProcessRequest,
    BatchRequestStatus,
    BatchUserAction,
    DataCollection,
    Geometry,
    MimeType,
    MosaickingOrder,
    ResamplingType,
    SentinelHubRequest,
    monitor_batch_process_analysis,
    monitor_batch_process_job,
)
from sentinelhub.api.utils import s3_specification
from sentinelhub.exceptions import DownloadFailedException

from eogrow.core.area.base import get_geometry_from_file, load_grid, save_grid
from eogrow.core.area.custom_grid import CustomGridAreaManager
from eogrow.core.area.utm import create_utm_zone_grid
from eogrow.utils.fs import LocalFile

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..types import TimePeriod
from ..utils.validators import (
    ensure_storage_key_presence,
    field_validator,
    optional_field_validator,
    parse_data_collection,
    parse_time_period,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
P = ParamSpec("P")


def _retry_on_404(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def retrying_func(*args: P.args, **kwargs: P.kwargs) -> T:
        for wait_time in [0, 10, 20, 100]:
            time.sleep(wait_time)  # if we start monitoring too soon we might hit a 404
            try:
                return func(*args, **kwargs)
            except DownloadFailedException as e:
                if (
                    e.request_exception is not None
                    and e.request_exception.response is not None
                    and e.request_exception.response.status_code == requests.status_codes.codes.NOT_FOUND
                ):
                    LOGGER.info("Received error 404 on monitoring endpoint. Retrying in a while.")
                    continue  # we retry on 404
                raise e

        time.sleep(wait_time)  # uses longest wait time from loop
        return func(*args, **kwargs)  # try one last time and fail explicitly

    return retrying_func


class InputDataSchema(BaseSchema):
    """Parameter structure for a single data collection used in a batch request."""

    data_collection: DataCollection = Field(
        description=(
            "Data collection from which data will be downloaded. See `utils.validators.parse_data_collection` for more"
            " info on input options."
        )
    )
    _validate_data_collection = field_validator("data_collection", parse_data_collection, pre=True)

    time_period: Optional[TimePeriod]
    _validate_time_period = optional_field_validator("time_period", parse_time_period, pre=True)

    resampling_type: ResamplingType = Field(
        ResamplingType.NEAREST, description="A type of downsampling and upsampling used by Sentinel Hub service"
    )
    maxcc: Optional[float] = Field(ge=0, le=1, description="Maximal cloud coverage filter.")
    mosaicking_order: Optional[MosaickingOrder] = Field(description="The mosaicking order used by Sentinel Hub service")
    other_params: dict = Field(
        default_factory=dict,
        description=(
            "Additional parameters to be passed to SentinelHubRequest.input_data method as other_args parameter."
        ),
    )


class BatchGridSchema(BaseSchema):
    """Configuration for the batch grid."""

    geometry_filename: str = Field(
        description="Name of the file that defines the AoI geometry, located in the input data folder."
    )
    bbox_size: tuple[int, int] = Field(description="Size of the bounding box in meters.")
    bbox_offset: tuple[float, float] = Field(description="Offset of the bounding box in meters.")
    bbox_buffer: tuple[float, float] = Field(description="Buffer of the bounding box in meters.")
    image_size: tuple[int, int] = Field(description="Size of the image in pixels.")
    resolution: int = Field(description="Resolution of the image in meters.")


class BatchDownloadPipeline(Pipeline):
    """
    Pipeline to start and monitor a Sentinel Hub Batch Process API job

    The pipeline creates a custom grid using the UtmZoneSplitter under the hood and saves it to the grid location
    provided via the CustomGridAreaManager.
    """

    NAME_COLUMN = "identifier"

    class Schema(Pipeline.Schema):
        area: CustomGridAreaManager.Schema

        @validator("area")
        def _parse_area_name_column(cls, area: CustomGridAreaManager.Schema) -> CustomGridAreaManager.Schema:
            assert area.name_column == BatchDownloadPipeline.NAME_COLUMN, (
                "Name column of CustomGridAreaManager used in BatchDownloadPipeline should be "
                f"set to '{BatchDownloadPipeline.NAME_COLUMN}' for proper functionality."
            )
            return area

        iam_role_arn: str = Field(description="IAM role ARN for the batch job.")

        output_folder_key: str = Field(
            description="Storage manager key pointing to the path where batch results will be saved."
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        grid: BatchGridSchema = Field(description="Configuration for the batch grid.")

        inputs: List[InputDataSchema]

        evalscript_folder_key: str = Field(
            "input_data", description="Storage manager key pointing to the path where the evalscript is loaded from."
        )
        _ensure_evalscript_folder_key = ensure_storage_key_presence("evalscript_folder_key")
        evalscript_path: str

        tiff_outputs: List[str] = Field(default_factory=list, description="Names of TIFF outputs of a batch job")
        save_userdata: bool = Field(
            False, description="A flag indicating if userdata.json should also be one of the results of the batch job."
        )
        batch_output_kwargs: dict = Field(
            default_factory=dict,
            description=(
                "Any other arguments to be added to a dictionary of parameters. Passed as `**kwargs` to the `output`"
                " method of `BatchProcessClient` during the creation process."
            ),
        )

        analysis_only: bool = Field(
            False,
            description=(
                "If set to True it will only create a batch request and wait for analysis phase to finish. It "
                "will not start the actual batch job."
            ),
        )
        monitoring_sleep_time: int = Field(
            120,
            ge=60,
            description=(
                "How many seconds to sleep between two consecutive queries about status of tiles in a batch "
                "job. It should be at least 60 seconds."
            ),
        )
        monitoring_analysis_sleep_time: int = Field(
            10,
            ge=5,
            description=(
                "How many seconds to sleep between two consecutive queries about a status of a batch job analysis "
                "phase. It should be at least 5 seconds."
            ),
        )

        batch_id: str = Field(
            "",
            description=(
                "An ID of a batch job for this pipeline. If it is given the pipeline will just monitor the "
                "existing batch job. If it is not given it will create a new batch job."
            ),
        )
        patch_list: None = None
        input_patch_file: None = None
        skip_existing: Literal[False] = False

    config: Schema
    area_manager: CustomGridAreaManager

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.batch_client = BatchProcessClient(config=self.sh_config)

    def run_procedure(self) -> tuple[list[str], list[str]]:
        """Procedure that uses Sentinel Hub batch service to download data to an S3 bucket."""
        batch_request = self._create_or_collect_batch_request()

        user_action = self._trigger_user_action(batch_request)
        self._wait_for_sh_db_sync(batch_request)

        if user_action is BatchUserAction.ANALYSE or (
            user_action is BatchUserAction.START and batch_request.status is not BatchRequestStatus.ANALYSIS_DONE
        ):
            LOGGER.info("Waiting to finish analyzing job with ID %s", batch_request.request_id)
            monitor_batch_process_analysis(
                request=batch_request,
                client=self.batch_client,
                sleep_time=self.config.monitoring_analysis_sleep_time,
            )

        if self.config.analysis_only:
            return [], []

        LOGGER.info("Monitoring batch job with ID %s", batch_request.request_id)
        batch_request = monitor_batch_process_job(
            request=batch_request,
            client=self.batch_client,
            sleep_time=self.config.monitoring_sleep_time,
            analysis_sleep_time=self.config.monitoring_analysis_sleep_time,
        )

        LOGGER.info("Using feature manifest to update the batch grid")
        self._update_batch_grid(batch_request.request_id)

        tiles_dict = self._get_tiles_per_status(batch_request.request_id)
        processed = tiles_dict.get("DONE", [])
        failed = tiles_dict.get("FATAL", [])

        log_msg = f"Successfully downloaded {len(processed)} tiles"
        log_msg += f", but {len(failed)} tiles failed." if failed else "."
        LOGGER.info(log_msg)
        return processed, failed

    def _create_or_collect_batch_request(self) -> BatchProcessRequest:
        """Either creates a new batch request or collects information about an existing one."""
        if not self.config.batch_id:
            batch_request = self._create_new_batch_request()
            LOGGER.info("Created a new batch request with ID %s", batch_request.request_id)
            return batch_request

        batch_request = self.batch_client.get_request(self.config.batch_id)
        batch_request.raise_for_status(status=BatchRequestStatus.FAILED)
        LOGGER.info("Collected existing batch request with ID %s", batch_request.request_id)
        return batch_request

    def _get_aoi_geometry(self) -> Geometry:
        """Gets the geometry from the input data folder."""
        geom_path = fs.path.join(self.storage.get_input_data_folder(), self.config.grid.geometry_filename)
        return get_geometry_from_file(
            filesystem=self.storage.filesystem,
            file_path=geom_path,
            geopandas_engine=self.storage.config.geopandas_backend,
        )

    def _create_and_save_batch_grid(self) -> str:
        """Creates a saves the grid used for Batch Process API"""
        grid = create_utm_zone_grid(
            geometry=self._get_aoi_geometry(),
            name_column=self.NAME_COLUMN,
            bbox_size=self.config.grid.bbox_size,
            bbox_offset=self.config.grid.bbox_offset,
            bbox_buffer=self.config.grid.bbox_buffer,
        )
        grid = to_batch_grid_format(grid, self.config.grid.image_size, self.config.grid.resolution)

        grid_folder = self.storage.get_folder(self.area_manager.config.grid_folder_key)
        grid_path = fs.path.join(grid_folder, self.area_manager.config.grid_filename)
        save_grid(grid, grid_path, self.storage)
        return grid_path

    def _update_batch_grid(self, batch_request_id: str) -> None:
        """Updates the batch grid using the features manifest."""
        grid_folder = self.storage.get_folder(self.area_manager.config.grid_folder_key)
        grid_path = fs.path.join(grid_folder, self.area_manager.config.grid_filename)
        grid = load_grid(grid_path, self.storage)

        fm_folder = self.storage.get_folder(self.config.output_folder_key)
        fm_path = fs.path.join(fm_folder, f"featureManifest-{batch_request_id}.gpkg")
        fm = load_grid(fm_path, self.storage)

        for crs, crs_grid in grid.items():
            grid[crs] = crs_grid[crs_grid[self.NAME_COLUMN].isin(fm[crs][self.NAME_COLUMN].unique())]

        save_grid(grid, grid_path, self.storage)

    def _create_new_batch_request(self) -> BatchProcessRequest:
        """Defines a new batch request."""
        geometry = self._get_aoi_geometry()
        grid_path = self._create_and_save_batch_grid()

        responses = [
            SentinelHubRequest.output_response(tiff_output, MimeType.TIFF) for tiff_output in self.config.tiff_outputs
        ]
        if self.config.save_userdata:
            responses.append(SentinelHubRequest.output_response("userdata", MimeType.JSON))

        sentinelhub_request = SentinelHubRequest(
            evalscript=self._get_evalscript(),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=input_config.data_collection,
                    time_interval=input_config.time_period,
                    upsampling=input_config.resampling_type,
                    downsampling=input_config.resampling_type,
                    maxcc=input_config.maxcc,
                    mosaicking_order=input_config.mosaicking_order,
                    other_args=input_config.other_params,
                )
                for input_config in self.config.inputs
            ],
            responses=responses,
            geometry=geometry,
        )

        data_folder = self.storage.get_folder(self.config.output_folder_key, full_path=True).rstrip("/")
        if not self.storage.is_on_s3():
            raise ValueError(f"The data folder path should be on s3 bucket, got {data_folder}")

        geopackage_input = BatchProcessClient.geopackage_input(
            s3_specification(url=grid_path, iam_role_arn=self.config.iam_role_arn)
        )

        raster_output = BatchProcessClient.raster_output(
            delivery=s3_specification(
                url=f"{data_folder}/<tileName>/<outputId>.<format>", iam_role_arn=self.config.iam_role_arn
            ),
            **self.config.batch_output_kwargs,
        )

        return self.batch_client.create(
            process_request=sentinelhub_request,
            input=geopackage_input,
            output=raster_output,
            description=f"eo-grow - {self.__class__.__name__} pipeline with ID {self.pipeline_id}",
        )

    def _get_evalscript(self) -> str:
        evalscript_path = fs.path.join(
            self.storage.get_folder(self.config.evalscript_folder_key), self.config.evalscript_path
        )
        with self.storage.filesystem.open(evalscript_path) as evalscript_file:
            return evalscript_file.read()

    @_retry_on_404
    def _trigger_user_action(self, batch_request: BatchProcessRequest) -> BatchUserAction:
        """According to status and configuration parameters decide what kind of user action to perform."""
        if self.config.analysis_only:
            if batch_request.status is BatchRequestStatus.CREATED:
                self.batch_client.start_analysis(batch_request)
                LOGGER.info("Triggered batch job analysis.")
                return BatchUserAction.ANALYSE

            status = None if batch_request.status is None else batch_request.status.value
            LOGGER.info("Didn't trigger analysis because current batch request status is %s.", status)
            return BatchUserAction.NONE

        if batch_request.status in [
            BatchRequestStatus.CREATED,
            BatchRequestStatus.ANALYSING,
            BatchRequestStatus.ANALYSIS_DONE,
        ]:
            self.batch_client.start_job(batch_request)
            LOGGER.info("Started running batch job.")
            return BatchUserAction.START

        status = None if batch_request.status is None else batch_request.status.value
        LOGGER.info("Didn't trigger batch job because current batch request status is %s", status)
        return BatchUserAction.NONE

    @_retry_on_404
    def _wait_for_sh_db_sync(self, batch_request: BatchProcessRequest) -> None:
        """Wait for SH read/write databases to sync."""
        self.batch_client.get_request(batch_request)

    def _get_tiles_per_status(self, batch_request_id: str) -> dict[str, list[str]]:
        """
        Collects tile status counts from the batch request execution sqlite database for the PENDING, DONE and FAILED
        statuses.

        DONE: Feature was successfully processed.
        FATAL: Feature has failed X amount of times and will not be retried.
        PENDING: The feature is waiting to be processed.
        """
        db_folder = self.storage.get_folder(self.area_manager.config.grid_folder_key)
        db_path = fs.path.join(db_folder, f"execution-{batch_request_id}.gpkg")
        with LocalFile(db_path, mode="r", filesystem=self.storage.filesystem) as local_file:
            conn = sqlite3.connect(local_file.path)
            db_df = pd.read_sql("SELECT * FROM features", conn)
            return db_df.groupby("status").name.apply(list).to_dict()


def to_batch_grid_format(
    grid: dict[CRS, GeoDataFrame], image_size: tuple[int, int], resolution: int
) -> dict[CRS, GeoDataFrame]:
    """Updates a grid to a format suitable for use with Batch Processing API."""
    width, height = image_size
    for crs, gdf in grid.items():
        gdf["width"] = width
        gdf["height"] = height
        gdf["resolution"] = resolution
        grid[crs] = gdf

    return grid
