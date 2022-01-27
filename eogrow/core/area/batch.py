"""
Area managers for Sentinel Hub batch grids
"""
from typing import List

from geopandas.geodataframe import GeoDataFrame
from pydantic import Field

from sentinelhub import BBox, SentinelHubBatch, BatchSplitter, BatchRequestStatus
from sentinelhub.sentinelhub_batch import BatchRequest

from .base import AreaManager


class BatchAreaManager(AreaManager):
    """Area manager that is based on Sentinel Hub Batch service tiling"""

    class Schema(AreaManager.Schema):
        """A schema used for pipelines based on batch tiling"""

        tiling_grid_id: int
        resolution: float
        tile_buffer: int = Field(0, description="Number of pixels for which each tile will be buffered")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_id = ""

    def _get_grid_filename_params(self) -> List[object]:
        """Puts together batch parameters"""
        return [self.config.tiling_grid_id, self.config.resolution, self.config.tile_buffer]

    def _create_new_split(self, *_, **__) -> List[GeoDataFrame]:
        """Instead of creating a split it loads tiles from the service"""
        if not self.batch_id:
            raise ValueError(
                "Trying to create a new batch grid but cannot collect tile geometries because 'batch_id' has not been "
                "given. Make sure that you are running a pipeline that creates a new batch job."
            )

        batch_request = SentinelHubBatch().get_request(self.batch_id)
        self._verify_batch_request(batch_request)

        splitter = BatchSplitter(batch_request=batch_request)
        bbox_list = splitter.get_bbox_list()
        info_list = splitter.get_info_list()

        if not all(bbox.crs.is_utm() for bbox in bbox_list):
            raise NotImplementedError("So far we only support UTM-based batch tiling grids")

        # The problem is that Sentinel Hub service returns unbuffered tile geometries in WGS84. BatchSplitter then
        # transforms them into UTM, but here we still have to fix numerical errors by rounding and then apply buffer.
        bbox_list = [self._fix_batch_bbox(bbox) for bbox in bbox_list]

        # Sentinel Hub service provides tile info in an arbitrary order. We sort by tile names for consistency.
        bbox_list, info_list = zip(*sorted(zip(bbox_list, info_list), key=lambda pair: pair[1]["name"]))
        for index, info in enumerate(info_list):
            info["index_n"] = index

        return self._to_dataframe_grid(bbox_list, info_list, ["index_n", "id", "name", "cost"])

    def _verify_batch_request(self, batch_request: BatchRequest):
        """Verifies that given batch request has finished and that it has the same tiling grid parameters as
        they are written in the config.
        """
        batch_request.raise_for_status(status=[BatchRequestStatus.FAILED, BatchRequestStatus.CANCELED])

        expected_tiling_grid_params = {
            "id": self.config.tiling_grid_id,
            "bufferY": self.config.tile_buffer,
            "bufferX": self.config.tile_buffer,
            "resolution": self.config.resolution,
        }
        if batch_request.tiling_grid != expected_tiling_grid_params:
            raise ValueError(
                f"Tiling grid parameters in config are {expected_tiling_grid_params} but given batch "
                f"request has parameters {batch_request.tiling_grid}"
            )

    def _fix_batch_bbox(self, bbox: BBox) -> BBox:
        """Fixes a batch tile bounding box so that it will be the same as in produced tiles on the bucket"""
        min_x, min_y, max_x, max_y = list(map(round, bbox))

        buffer_meters = self.config.tile_buffer * self.config.resolution

        min_x, min_y = min_x - buffer_meters, min_y - buffer_meters
        max_x, max_y = max_x + buffer_meters, max_y + buffer_meters

        return BBox((min_x, min_y, max_x, max_y), crs=bbox.crs)
