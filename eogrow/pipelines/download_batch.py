"""
Download pipeline that works with Sentinel Hub batch service
"""
import logging
from typing import List, Optional, Tuple, Dict, DefaultDict

from pydantic import Field, conint

from sentinelhub import (
    SentinelHubRequest,
    SentinelHubBatch,
    BatchRequest,
    MimeType,
    BatchRequestStatus,
    BatchTileStatus,
    monitor_batch_job,
    read_data,
    DataCollection,
)

from ..core.pipeline import Pipeline
from ..utils.validators import (
    field_validator,
    optional_field_validator,
    validate_mosaicking_order,
    validate_resampling,
    parse_time_period,
    parse_data_collection,
)
from ..utils.types import Path, TimePeriod

LOGGER = logging.getLogger(__name__)


class BatchDownloadPipeline(Pipeline):
    """Pipeline to start and monitor a Sentinel Hub batch job"""

    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(
            description="Storage manager key pointing to the path where batch results will be saved."
        )
        data_collection: DataCollection = Field(description="Data collection from which data will be downloaded.")
        _validate_data_collection = field_validator("data_collection", parse_data_collection, pre=True)

        time_period: TimePeriod
        _validate_time_period = field_validator("time_period", parse_time_period, pre=True)

        evalscript_path: Path
        tiff_outputs: List[str] = Field(default_factory=list, description="Names of TIFF outputs of a batch job")
        save_userdata: bool = Field(
            False, description="A flag indicating if userdata.json should also be oneof the results of the batch job."
        )

        resampling_type: str = Field(
            "NEAREST", description="A type of downsampling and upsampling used by Sentinel Hub service"
        )
        _validate_resampling_type = field_validator("resampling_type", validate_resampling)

        mosaicking_order: Optional[str] = Field(description="The mosaicking order used by Sentinel Hub service")
        _validate_mosaicking_order = optional_field_validator("mosaicking_order", validate_mosaicking_order)

        monitoring_sleep_time: conint(ge=60) = Field(
            120,
            description=(
                "How many seconds to sleep between two consecutive queries about status of tiles in a batch "
                "job. It should be at least 60 seconds."
            ),
        )

        batch_id: str = Field(
            "",
            description=(
                "An ID of a batch job for this pipeline. If it is given the pipeline will just monitor the "
                "existing batch job. If it is not given it will create a new batch job."
            ),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_client = SentinelHubBatch(config=self.sh_config)

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Procedure that downloads data to an s3 bucket using batch service"""
        batch_request = self._prepare_and_run_batch_request()
        self.area_manager.batch_id = batch_request.request_id

        LOGGER.info("Monitoring batch job with ID %s", batch_request.request_id)
        results = monitor_batch_job(batch_request, config=self.sh_config, sleep_time=self.config.monitoring_sleep_time)

        if batch_request.value_estimate:
            LOGGER.info("Estimated cost of the batch job: %.1f", batch_request.value_estimate)

        processed = self._get_tile_names_from_results(results, BatchTileStatus.PROCESSED)
        failed = self._get_tile_names_from_results(results, BatchTileStatus.FAILED)
        return processed, failed

    def _prepare_and_run_batch_request(self) -> BatchRequest:
        """Either creates a new job and starts running it or restarts partially failed job or just reports batch
        request status.
        """
        if not self.config.batch_id:
            batch_request = self._create_new_batch_request()
            self.batch_client.start_job(batch_request)
            LOGGER.info("Created a new batch job with ID %s", batch_request.request_id)
            return batch_request

        batch_request = self.batch_client.get_request(self.config.batch_id)
        batch_request.raise_for_status(status=[BatchRequestStatus.FAILED, BatchRequestStatus.CANCELED])

        if batch_request.status is BatchRequestStatus.PARTIAL:
            self.batch_client.restart_job(batch_request)
            LOGGER.info("Restarted batch job %s", batch_request.request_id)
        else:
            LOGGER.info("Current request status is %s", batch_request.status.value)

        return batch_request

    def _create_new_batch_request(self) -> BatchRequest:
        """Defines a new batch request"""
        geometry = self.area_manager.get_area_geometry()

        responses = [
            SentinelHubRequest.output_response(tiff_output, MimeType.TIFF) for tiff_output in self.config.tiff_outputs
        ]
        if self.config.save_userdata:
            responses.append(SentinelHubRequest.output_response("userdata", MimeType.JSON))

        sentinelhub_request = SentinelHubRequest(
            evalscript=read_data(self.config.evalscript_path, data_format=MimeType.TXT),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.config.data_collection,
                    time_interval=self.config.time_period,
                    upsampling=self.config.resampling_type,
                    downsampling=self.config.resampling_type,
                    mosaicking_order=self.config.mosaicking_order,
                )
            ],
            responses=responses,
            geometry=geometry,
        )

        data_folder = self.storage.get_folder(self.config.output_folder_key, full_path=True).rstrip("/")
        if not self.storage.is_on_aws():
            raise ValueError(f"The data folder path should be on s3 bucket, got {data_folder}")

        return self.batch_client.create(
            sentinelhub_request,
            tiling_grid=SentinelHubBatch.tiling_grid(
                grid_id=self.config.area.tiling_grid_id,
                resolution=self.config.area.resolution,
                buffer=(self.config.area.tile_buffer, self.config.area.tile_buffer),
            ),
            output=SentinelHubBatch.output(default_tile_path=f"{data_folder}/<tileName>/<outputId>.<format>"),
            description=f"eo-grow - {self.__class__.__name__} pipeline with ID {self.pipeline_id}",
        )

    @staticmethod
    def _get_tile_names_from_results(results: DefaultDict[str, List[Dict]], tile_status: BatchTileStatus) -> List[str]:
        """Collects tile names from a dictionary of batch tile results ordered by status"""
        tile_list = results[tile_status.value]
        return [tile["name"] for tile in tile_list]
