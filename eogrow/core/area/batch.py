"""
Area managers for Sentinel Hub batch grids
"""
from typing import Any, List, Tuple

from geopandas import GeoDataFrame
from pydantic import Field

from sentinelhub import BatchRequest, BatchRequestStatus, BatchSplitter, BBox, SentinelHubBatch

from ...utils.general import convert_bbox_coords_to_int
from .base import AreaManager


class BatchAreaManager(AreaManager):
    """Area manager that is based on Sentinel Hub Batch service tiling"""

    class Schema(AreaManager.Schema):
        """A schema used for pipelines based on batch tiling"""

        tiling_grid_id: int = Field(
            description="An id of one of the tiling grids predefined at Sentinel Hub Batch service."
        )
        resolution: float = Field(
            description=(
                "One of the resolutions that are predefined at Sentinel Hub Batch service for chosen tiling_grid_id."
            )
        )
        tile_buffer_x: int = Field(0, description="Number of pixels for which to buffer each tile left and right.")
        tile_buffer_y: int = Field(0, description="Number of pixels for which to buffer each tile up and down.")
        batch_id: str = Field(
            "",
            description=(
                "An ID of a batch job for this pipeline. If it is given the pipeline will just monitor the "
                "existing batch job. If it is not given it will create a new batch job."
            ),
        )
        subsplit_x: int = Field(
            1, ge=1, description="Number of sub-tiles into which each batch tile is split along horizontal dimension"
        )
        subsplit_y: int = Field(
            1, ge=1, description="Number of sub-tiles into which each batch tile is split along vertical dimension"
        )

    config: Schema
    _BATCH_GRID_COLUMNS = ["index_n", "id", "name", "split_x", "split_y"]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.batch_id = self.config.batch_id

        self.subsplit = self.config.subsplit_x, self.config.subsplit_y
        self.absolute_buffer = (
            self.config.tile_buffer_x * self.config.resolution,
            self.config.tile_buffer_y * self.config.resolution,
        )

    def _get_grid_filename_params(self) -> List[object]:
        """Puts together batch parameters"""
        params: List[Any] = [
            self.config.tiling_grid_id,
            self.config.resolution,
            self.config.tile_buffer_x,
            self.config.tile_buffer_y,
        ]
        if self.subsplit != (1, 1):
            params.extend([self.config.subsplit_x, self.config.subsplit_y])
        return params

    def _create_new_split(self, *_: Any, **__: Any) -> List[GeoDataFrame]:
        """Instead of creating a split it loads tiles from the service"""
        if not self.batch_id:
            raise MissingBatchIdError(
                "Trying to create a new batch grid but cannot collect tile geometries because 'batch_id' has not been "
                f"given. You can either define it in {self.__class__.__name__} config schema or run a pipeline that "
                "creates a new batch request."
            )

        batch_client = SentinelHubBatch(config=self.storage.sh_config)
        batch_request = batch_client.get_request(self.batch_id)
        self._verify_batch_request(batch_request)

        splitter = BatchSplitter(batch_request=batch_request, config=self.storage.sh_config)
        bbox_list = splitter.get_bbox_list()
        info_list = splitter.get_info_list()

        if not all(bbox.crs.is_utm() for bbox in bbox_list):
            raise NotImplementedError("So far we only support UTM-based batch tiling grids")

        # The problem is that Sentinel Hub service returns unbuffered tile geometries in WGS84. BatchSplitter then
        # transforms them into UTM, but here we still have to fix numerical errors by rounding and then apply buffer.
        bbox_list = [self._fix_batch_bbox(bbox) for bbox in bbox_list]

        for info in info_list:
            info["split_x"] = 0
            info["split_y"] = 0

        return self._to_dataframe_grid(bbox_list, info_list, self._BATCH_GRID_COLUMNS)

    def _verify_batch_request(self, batch_request: BatchRequest) -> None:
        """Verifies that given batch request has finished and that it has the same tiling grid parameters as
        they are written in the config.
        """
        batch_request.raise_for_status(status=[BatchRequestStatus.FAILED, BatchRequestStatus.CANCELED])

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

    def _fix_batch_bbox(self, bbox: BBox) -> BBox:
        """Fixes a batch tile bounding box so that it will be the same as in produced tiles on the bucket."""
        corrected_bbox = convert_bbox_coords_to_int(bbox)
        return corrected_bbox.buffer(self.absolute_buffer, relative=False)

    def _to_dataframe_grid(
        self, bbox_list: List[BBox], info_list: List[dict], info_columns: List[str]
    ) -> List[GeoDataFrame]:
        """Sentinel Hub service provides tile info in an arbitrary order. This sorts by tile name and indices for
        consistency. Then it converts them into dataframe grid.
        """

        def sort_key_function(values: Tuple[Any, dict]) -> Tuple[str, int, int]:
            _, info = values
            return info["name"], info["split_x"], info["split_y"]

        bbox_list, info_list = zip(*sorted(zip(bbox_list, info_list), key=sort_key_function))
        for index, info_dict in enumerate(info_list):
            info_dict["index_n"] = index

        return super()._to_dataframe_grid(bbox_list, info_list, info_columns)


class MissingBatchIdError(ValueError):
    """Exception that is triggered if ID of a Sentinel Hub batch job is missing."""

