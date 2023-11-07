"""Area manager implementation for Sentinel Hub batch grids."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import fs
import geopandas as gpd
from geopandas import GeoDataFrame
from pydantic import Field

from sentinelhub import CRS, BatchRequest, BatchRequestStatus, BatchSplitter, Geometry, SentinelHubBatch

from ..storage import StorageManager
from .base import BaseAreaManager, get_geometry_from_file

LOGGER = logging.getLogger(__name__)


class MissingBatchIdError(ValueError):
    """Exception that is triggered if ID of a Sentinel Hub batch job is missing."""


class BatchAreaManager(BaseAreaManager):
    """Area manager that splits the area according to the Sentinel Hub Batch tiling grids."""

    class Schema(BaseAreaManager.Schema):
        geometry_filename: str = Field(
            description="Name of the file that defines the AoI geometry, located in the input data folder."
        )
        tiling_grid_id: int = Field(
            description="An id of one of the tiling grids predefined at Sentinel Hub Batch service."
        )
        resolution: float = Field(
            description=(
                "Resolution of downloaded imagery in meters. Resolution options are predefined at Sentinel Hub Batch"
                " service for a chosen tiling_grid_id."
            )
        )
        tile_buffer_x: int = Field(0, description="Number of pixels for which to buffer each tile left and right.")
        tile_buffer_y: int = Field(0, description="Number of pixels for which to buffer each tile up and down.")
        batch_id: Optional[str] = Field(
            description=(
                "ID of the batch job that defines the AOI. Not required when using BatchDownloadPipeline,"
                " which generates a new batch job with the given AOI parameters."
            ),
        )

    config: Schema

    def __init__(self, config: Schema, storage: StorageManager):
        super().__init__(config, storage)
        # We provide a way to inject a Batch ID after initialization if no ID was given in the config
        # This is meant to be used only in the BatchDownloadPipeline to force caching
        self._injected_batch_id: str | None = None

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84) -> Geometry:
        file_path = fs.path.join(self.storage.get_input_data_folder(), self.config.geometry_filename)
        return get_geometry_from_file(
            filesystem=self.storage.filesystem,
            file_path=file_path,
            geopandas_engine=self.storage.config.geopandas_backend,
        ).transform(crs)

    def _create_grid(self) -> dict[CRS, GeoDataFrame]:
        """Uses BatchSplitter to create a grid for the selected batch job."""
        batch_id = self.config.batch_id or self._injected_batch_id

        if batch_id is None:
            raise MissingBatchIdError(
                "Trying to create a new batch grid but cannot collect tile geometries because 'batch_id' has not been "
                f"given. You can either provide it in the {self.__class__.__name__} schema or run a pipeline that "
                "creates a new batch request."
            )

        batch_client = SentinelHubBatch(config=self.storage.sh_config)
        batch_request = batch_client.get_request(batch_id)
        self._verify_batch_request(batch_request)

        splitter = BatchSplitter(batch_request=batch_request, config=self.storage.sh_config)
        bbox_list, info_list = splitter.get_bbox_list(), splitter.get_info_list()

        crs_to_patches = defaultdict(list)
        # they are returned in random order, so we sort them by name beforehand
        for bbox, info in sorted(zip(bbox_list, info_list), key=lambda x: x[1]["name"]):  # type: ignore # noqa: PGH003
            crs_to_patches[bbox.crs].append((info["name"], bbox.geometry))

        grid = {}
        for crs, named_bbox_geoms in crs_to_patches.items():
            names, geoms = zip(*named_bbox_geoms)
            grid[crs] = gpd.GeoDataFrame({self.NAME_COLUMN: names}, geometry=list(geoms), crs=crs.pyproj_crs())

        return grid

    def _verify_batch_request(self, batch_request: BatchRequest) -> None:
        """Verifies that the given batch request has finished and that it contains the same tiling grid parameters as
        in the config.
        """
        batch_request.raise_for_status(
            status=(
                BatchRequestStatus.CREATED,  # tiles not available prior to analysis
                BatchRequestStatus.ANALYSING,  # tiles not available prior to analysis
                BatchRequestStatus.FAILED,
                BatchRequestStatus.CANCELED,
            )
        )

        expected_tiling_grid_params = {
            "id": self.config.tiling_grid_id,
            "bufferY": self.config.tile_buffer_y,
            "bufferX": self.config.tile_buffer_x,
            "resolution": self.config.resolution,
        }
        if batch_request.tiling_grid != expected_tiling_grid_params:
            raise ValueError(
                f"Tiling grid parameters in config are {expected_tiling_grid_params} but given batch "
                f"request has parameters {batch_request.tiling_grid}"
            )

    def get_grid_cache_filename(self) -> str:
        input_filename = fs.path.basename(self.config.geometry_filename)
        input_filename = input_filename.rsplit(".", 1)[0]

        raw_params = [
            input_filename,
            self.config.tiling_grid_id,
            self.config.resolution,
            self.config.tile_buffer_x,
            self.config.tile_buffer_y,
        ]
        params = [str(param) for param in raw_params]

        return f"{self.__class__.__name__}_{'_'.join(params)}.gpkg"
