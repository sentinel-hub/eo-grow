"""
Area managers for Sentinel Hub batch grids
"""
import warnings
from typing import Any, List, Tuple, cast

from geopandas import GeoDataFrame
from pydantic import Field

from eolearn.core.exceptions import EODeprecationWarning
from sentinelhub import CRS, BatchRequest, BatchRequestStatus, BatchSplitter, BBox, SentinelHubBatch

from ...utils.general import convert_bbox_coords_to_int, reduce_to_coprime
from ...utils.grid import GridTransformation, create_transformations, get_grid_bbox
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
        subsplit_x: int = Field(
            1, ge=1, description="Number of sub-tiles into which each batch tile is split along horizontal dimension."
        )
        subsplit_y: int = Field(
            1, ge=1, description="Number of sub-tiles into which each batch tile is split along vertical dimension."
        )

        batch_id: str = Field(
            "",
            description=(
                "An ID of a batch job for this pipeline. If it is given the pipeline will just monitor the "
                "existing batch job. If it is not given it will create a new batch job."
            ),
        )

    config: Schema
    _BATCH_GRID_COLUMNS = ["index_n", "id", "name", "split_x", "split_y"]
    _SH_REPROJECTION_ERROR = 1e-3  # Rounding happens in Sentinel Hub Batch database where coordinates are in WGS84

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.batch_id = self.config.batch_id

        self.subsplit = self.config.subsplit_x, self.config.subsplit_y
        self.absolute_buffer = (
            self.config.tile_buffer_x * self.config.resolution,
            self.config.tile_buffer_y * self.config.resolution,
        )

    def _get_grid_filename_params(self) -> List[object]:
        """Puts together batch parameters used for grid filename."""
        params: List[object] = [
            self.config.tiling_grid_id,
            self.config.resolution,
            self.config.tile_buffer_x,
            self.config.tile_buffer_y,
        ]
        if self.subsplit != (1, 1):
            params.extend([self.config.subsplit_x, self.config.subsplit_y])
        return params

    def _create_new_split(self, *_: Any, **__: Any) -> List[GeoDataFrame]:
        """Instead of creating a split it loads tiles from the service."""
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
        corrected_bbox = convert_bbox_coords_to_int(bbox, error=self._SH_REPROJECTION_ERROR)
        return corrected_bbox.buffer(self.absolute_buffer, relative=False)

    @staticmethod
    def _to_dataframe_grid(bbox_list: List[BBox], info_list: List[dict], info_columns: List[str]) -> List[GeoDataFrame]:
        """Sentinel Hub service provides tile info in an arbitrary order. This sorts by tile name and indices for
        consistency. Then it converts them into dataframe grid.
        """

        def sort_key_function(values: Tuple[Any, dict]) -> Tuple[str, int, int]:
            _, info = values
            return info["name"], info["split_x"], info["split_y"]

        bbox_list, info_list = zip(*sorted(zip(bbox_list, info_list), key=sort_key_function))
        for index, info_dict in enumerate(info_list):
            info_dict["index_n"] = index

        return AreaManager._to_dataframe_grid(bbox_list, info_list, info_columns)

    def transform_grid(self, target_area_manager: AreaManager) -> List[GridTransformation]:
        """Calculates how grid should be transformed into another area manager."""
        self._check_target_compatibility(target_area_manager)
        target_area_manager = cast(BatchAreaManager, target_area_manager)

        source_grid = self.get_grid(add_bbox_column=True)
        source_grid = _fix_split_columns(source_grid)
        target_grid = self._get_target_grid(target_area_manager, source_grid)

        subsplits_x, subsplits_y = zip(self.subsplit, target_area_manager.subsplit)
        source_shape, target_shape = zip(reduce_to_coprime(*subsplits_x), reduce_to_coprime(*subsplits_y))

        source_grid = self._add_patch_positions(source_grid, source_shape)
        target_grid = self._add_patch_positions(target_grid, target_shape)

        return create_transformations(source_grid, target_grid, match_columns=["name", "position_x", "position_y"])

    @staticmethod
    def _add_patch_positions(grid: List[GeoDataFrame], shape: Tuple[int, int]) -> List[GeoDataFrame]:
        """Adds columns of values that define how sub-tiles should be spatially divided into groups. Members of each
        group will have the same values in the newly added columns and will spatially form a regular subgrid of given
        shape."""
        for gdf in grid:
            gdf["position_x"] = gdf.split_x // shape[0]
            gdf["position_y"] = gdf.split_y // shape[1]

        return grid

    def _check_target_compatibility(self, target_area_manager: AreaManager) -> None:
        """Checks if source and target managers are compatible to define a transformation between them."""
        if not isinstance(target_area_manager, BatchAreaManager):
            raise NotImplementedError(
                f"Grid transformation is only supported into another {self.__class__.__name__}, but got "
                f"{target_area_manager.__class__.__name__}"
            )

        for param in ["tiling_grid_id", "resolution"]:
            if getattr(self.config, param) != getattr(target_area_manager.config, param):
                raise NotImplementedError(
                    "Cannot transform grid because source and target area managers have different values of config "
                    f"parameter {param}."
                )

    def _get_target_grid(
        self, target_area_manager: "BatchAreaManager", source_grid: List[GeoDataFrame]
    ) -> List[GeoDataFrame]:
        """First, it tries to obtain a target grid from target area manager. That would work if the grid is already
        cached or if target area manager is able to generate it. If not, then it creates the target grid from the
        given source grid."""
        try:
            return target_area_manager.get_grid(add_bbox_column=True)
        except MissingBatchIdError:
            pass

        basic_grid = self._join_batch_grid(source_grid, self.subsplit, self.absolute_buffer)
        target_grid = self._split_batch_grid(
            basic_grid, target_area_manager.subsplit, target_area_manager.absolute_buffer
        )
        if target_area_manager.subsplit != (1, 1):
            target_grid = self._filter_grid_with_area(target_grid)

        # Bounding boxes have to be removed before caching into a Geopackage because they are not serializable.
        target_grid_without_bbox = [gdf.drop(columns=["BBOX"]) for gdf in target_grid]
        target_area_manager.cache_grid(target_grid_without_bbox)

        return target_grid

    @staticmethod
    def _join_batch_grid(
        grid: List[GeoDataFrame], split: Tuple[int, int], buffer: Tuple[float, float]
    ) -> List[GeoDataFrame]:
        """Removes buffers and joins sub-tiles from a batch grid into original tiles."""
        reversed_buffer = -buffer[0], -buffer[1]

        joined_grid = []
        for gdf in grid:
            # A single bbox is enough to reconstruct the grid tile bbox.
            reduced_gdf = gdf.groupby(by="name").first().reset_index()

            reduced_gdf.BBOX = reduced_gdf.BBOX.apply(lambda bbox: bbox.buffer(reversed_buffer, relative=False))
            reduced_gdf.BBOX = reduced_gdf.filter(["BBOX", "split_x", "split_y"]).apply(
                lambda row: get_grid_bbox(row.BBOX, (row.split_x, row.split_y), split), axis=1
            )

            # Performs a numerical correction because it assumes the base batch grid is in UTM and all coordinates are
            # aligned to integers.
            reduced_gdf.BBOX = reduced_gdf.BBOX.apply(convert_bbox_coords_to_int)

            joined_grid.append(reduced_gdf)

        return joined_grid

    def _split_batch_grid(
        self, grid: List[GeoDataFrame], split: Tuple[int, int], buffer: Tuple[float, float]
    ) -> List[GeoDataFrame]:
        """Splits tiles of batch grid into sub-tiles and applies buffer."""
        new_tiles = []
        for gdf in grid:
            for _, row in gdf.iterrows():
                tile_bbox: BBox = row.BBOX
                partition = tile_bbox.get_partition(num_x=split[0], num_y=split[1])

                for index_x, bbox_list in enumerate(partition):
                    for index_y, bbox in enumerate(bbox_list):
                        tile_info = {
                            "name": row["name"],
                            "id": row["id"],
                            "split_x": index_x,
                            "split_y": index_y,
                            "BBOX": bbox.buffer(buffer, relative=False),
                        }
                        new_tiles.append(tile_info)

        bbox_list = [item["BBOX"] for item in new_tiles]
        return self._to_dataframe_grid(bbox_list, new_tiles, self._BATCH_GRID_COLUMNS + ["BBOX"])

    def _filter_grid_with_area(self, grid: List[GeoDataFrame]) -> List[GeoDataFrame]:
        """Removes those (sub)tiles in the grid that don't intersect with the area of interest."""
        area_geometry = self.get_area_geometry(ignore_region_filter=True)

        filtered_grid = []
        for gdf in grid:
            transformed_area_geometry = area_geometry.transform(CRS(gdf.crs))
            gdf = gdf[gdf.intersects(transformed_area_geometry.geometry)]
            filtered_grid.append(gdf)

        return filtered_grid


class MissingBatchIdError(ValueError):
    """Exception that is triggered if ID of a Sentinel Hub batch job is missing."""


def _fix_split_columns(grid: List[GeoDataFrame]) -> List[GeoDataFrame]:
    """This is temporary fix for handling batch grids that were cached with older eo-grow version."""
    columns_missing = False
    for gdf in grid:
        for column_name in ["split_x", "split_y"]:
            if column_name not in gdf:
                gdf[column_name] = 0
                columns_missing = True

    if columns_missing:
        warnings.warn(
            "Since eo-grow 1.1.0 cached batch grid should also have columns split_x and split_y. "
            "For the future, make sure to clear cached grid file and reproduce it.",
            category=EODeprecationWarning,
        )

    return grid
