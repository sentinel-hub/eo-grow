"""Download pipeline that works with Sentinel Hub batch service."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, List, Literal, Optional

import fs
from pydantic import Field

from sentinelhub import (
    BatchRequest,
    BatchRequestStatus,
    BatchTileStatus,
    BatchUserAction,
    DataCollection,
    MimeType,
    MosaickingOrder,
    ResamplingType,
    SentinelHubBatch,
    SentinelHubRequest,
    monitor_batch_analysis,
    monitor_batch_job,
)

from ..core.area.batch import BatchAreaManager
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


class BatchDownloadPipeline(Pipeline):
    """Pipeline to start and monitor a Sentinel Hub batch job"""

    class Schema(Pipeline.Schema):
        area: BatchAreaManager.Schema

        output_folder_key: str = Field(
            description="Storage manager key pointing to the path where batch results will be saved."
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

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
                " method of `SentinelHubBatch` during the creation process."
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
    area_manager: BatchAreaManager

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.batch_client = SentinelHubBatch(config=self.sh_config)

    def run_procedure(self) -> tuple[list[str], list[str]]:
        """Procedure that uses Sentinel Hub batch service to download data to an S3 bucket."""
        batch_request = self._create_or_collect_batch_request()

        user_action = self._trigger_user_action(batch_request)

        if user_action is BatchUserAction.ANALYSE or (
            user_action is BatchUserAction.START and batch_request.status is not BatchRequestStatus.ANALYSIS_DONE
        ):
            LOGGER.info("Waiting to finish analyzing job with ID %s", batch_request.request_id)
            monitor_batch_analysis(
                batch_request,
                config=self.sh_config,
                sleep_time=self.config.monitoring_analysis_sleep_time,
            )

        self.cache_batch_area_manager_grid(batch_request.request_id)

        if self.config.analysis_only:
            return [], []

        LOGGER.info("Monitoring batch job with ID %s", batch_request.request_id)
        results = monitor_batch_job(
            batch_request,
            config=self.sh_config,
            sleep_time=self.config.monitoring_sleep_time,
            analysis_sleep_time=self.config.monitoring_analysis_sleep_time,
        )

        processed = self._get_tile_names_from_results(results, BatchTileStatus.PROCESSED)
        failed = self._get_tile_names_from_results(results, BatchTileStatus.FAILED)
        log_msg = f"Successfully downloaded {len(processed)} tiles"
        log_msg += f", but {len(failed)} tiles failed." if failed else "."
        LOGGER.info(log_msg)
        return processed, failed

    def _create_or_collect_batch_request(self) -> BatchRequest:
        """Either creates a new batch request or collects information about an existing one."""
        if not self.config.batch_id:
            batch_request = self._create_new_batch_request()
            LOGGER.info("Created a new batch request with ID %s", batch_request.request_id)
            return batch_request

        batch_request = self.batch_client.get_request(self.config.batch_id)
        batch_request.raise_for_status(status=[BatchRequestStatus.FAILED, BatchRequestStatus.CANCELED])
        LOGGER.info("Collected existing batch request with ID %s", batch_request.request_id)
        return batch_request

    def _create_new_batch_request(self) -> BatchRequest:
        """Defines a new batch request."""
        geometry = self.area_manager.get_area_geometry()

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

        return self.batch_client.create(
            sentinelhub_request,
            tiling_grid=SentinelHubBatch.tiling_grid(
                grid_id=self.config.area.tiling_grid_id,
                resolution=self.config.area.resolution,
                buffer=(self.config.area.tile_buffer_x, self.config.area.tile_buffer_y),
            ),
            output=SentinelHubBatch.output(
                default_tile_path=f"{data_folder}/<tileName>/<outputId>.<format>", **self.config.batch_output_kwargs
            ),
            description=f"eo-grow - {self.__class__.__name__} pipeline with ID {self.pipeline_id}",
        )

    def _get_evalscript(self) -> str:
        evalscript_path = fs.path.join(
            self.storage.get_folder(self.config.evalscript_folder_key), self.config.evalscript_path
        )
        with self.storage.filesystem.open(evalscript_path) as evalscript_file:
            return evalscript_file.read()

    def _trigger_user_action(self, batch_request: BatchRequest) -> BatchUserAction:
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

        if batch_request.status is BatchRequestStatus.PARTIAL:
            self.batch_client.restart_job(batch_request)
            LOGGER.info("Restarted partially failed batch job.")
            return BatchUserAction.START

        status = None if batch_request.status is None else batch_request.status.value
        LOGGER.info("Didn't trigger batch job because current batch request status is %s", status)
        return BatchUserAction.NONE

    def cache_batch_area_manager_grid(self, request_id: str) -> None:
        """This method ensures that area manager caches batch grid into the storage."""
        if self.area_manager.config.batch_id and self.area_manager.config.batch_id != request_id:
            raise ValueError(
                f"{self.area_manager.__class__.__name__} is set to use batch request with ID "
                f"{self.area_manager.config.batch_id} but {self.__class__.__name__} is using batch request with ID "
                f"{request_id}. Make sure that you use the same IDs."
            )
        self.area_manager._injected_batch_id = request_id  # noqa: SLF001

        self.area_manager.get_grid()  # this caches the grid for later use

    @staticmethod
    def _get_tile_names_from_results(
        results: defaultdict[BatchTileStatus, list[dict]], tile_status: BatchTileStatus
    ) -> list[str]:
        """Collects tile names from a dictionary of batch tile results ordered by status"""
        tile_list = results[tile_status]
        return [tile["name"] for tile in tile_list]
