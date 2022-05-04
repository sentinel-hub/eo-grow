"""
Area managers working with UTM CRS
"""
import logging
from typing import Any, List, Tuple, cast

from geopandas import GeoDataFrame
from pydantic import Field

from sentinelhub import Geometry, UtmZoneSplitter

from ...utils.general import reduce_to_coprime
from ...utils.grid import GridTransformation, create_transformations
from .base import AreaManager

LOGGER = logging.getLogger(__name__)


class UtmZoneAreaManager(AreaManager):
    """Area manager that splits grid per UTM zones"""

    class Schema(AreaManager.Schema):
        patch_size_x: int = Field(description="A width of each EOPatch in meters")
        patch_size_y: int = Field(description="A height of each EOPatch in meters")
        patch_buffer_x: float = Field(
            0, description="Number of meters by which to increase the tile size to left and right."
        )
        patch_buffer_y: float = Field(
            0, description="Number of meters by which to increase the tile size to up and down."
        )
        offset_x: float = Field(0, description="An offset of tiling grid in horizontal dimension")
        offset_y: float = Field(0, description="An offset of tiling grid in vertical dimension")

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.absolute_buffer = self.config.patch_buffer_x, self.config.patch_buffer_y

    def _get_grid_filename_params(self) -> List[object]:
        """A list of parameters specific to UTM zone splitting"""
        return [
            self.config.patch_size_x,
            self.config.patch_size_y,
            self.config.patch_buffer_x,
            self.config.patch_buffer_y,
            self.config.offset_x,
            self.config.offset_y,
        ]

    def _create_new_split(self, area_geometry: Geometry) -> List[GeoDataFrame]:
        """Uses UtmZoneSplitter to create a grid"""
        LOGGER.info("Splitting area geometry into UTM zone grid")
        splitter = UtmZoneSplitter(
            [area_geometry.geometry],
            crs=area_geometry.crs,
            bbox_size=(self.config.patch_size_x, self.config.patch_size_y),
            offset=(self.config.offset_x, self.config.offset_y),
        )

        bbox_list = splitter.get_bbox_list()
        if self.absolute_buffer != (0, 0):
            bbox_list = [bbox.buffer(self.absolute_buffer, relative=False) for bbox in bbox_list]

        info_list = splitter.get_info_list()
        for tile_info in info_list:
            tile_info["index_n"] = tile_info["index"]
            tile_info["total_num"] = len(bbox_list)

        return self._to_dataframe_grid(bbox_list, info_list, ["index_n", "index_x", "index_y", "total_num"])

    def transform_grid(self, target_area_manager: AreaManager) -> List[GridTransformation]:
        """Calculates how grid should be transformed into another area manager."""
        self._check_target_compatibility(target_area_manager)
        target_area_manager = cast(UtmZoneAreaManager, target_area_manager)

        source_grid = self.get_grid(add_bbox_column=True)
        target_grid = target_area_manager.get_grid(add_bbox_column=True)

        source_size = self.config.patch_size_x, self.config.patch_size_y
        target_size = target_area_manager.config.patch_size_x, target_area_manager.config.patch_size_y

        sizes_x, sizes_y = zip(source_size, target_size)
        target_shape, source_shape = zip(reduce_to_coprime(*sizes_x), reduce_to_coprime(*sizes_y))

        source_grid = self._add_patch_positions(source_grid, self.config, source_shape)
        target_grid = self._add_patch_positions(target_grid, target_area_manager.config, target_shape)

        return create_transformations(source_grid, target_grid, match_columns=["position_x", "position_y"])

    @staticmethod
    def _add_patch_positions(grid: List[GeoDataFrame], config: Schema, shape: Tuple[int, int]) -> List[GeoDataFrame]:
        """Adds columns of values that define how sub-tiles should be spatially divided into groups. Members of each
        group will have the same values in the newly added columns and will spatially form a regular subgrid of given
        shape.

        The process works with the lower left vertex of the bounding box. First it reverts buffer and offset shift.
        This way each vertex will have coordinates that are an integer multiple of patch size because this is the way
        `UtmZoneSplitter` created the grid. In the next step the vertex values are divided by patch size and the given
        shape of the subgroup. This produces exactly the desired indices that define groups.
        """
        translate_x = config.patch_buffer_x - config.offset_x
        translate_y = config.patch_buffer_y - config.offset_y
        scale_x = config.patch_size_x * shape[0]
        scale_y = config.patch_size_y * shape[1]

        for gdf in grid:
            gdf["position_x"] = gdf.BBOX.apply(lambda bbox: round(bbox.min_x + translate_x) // scale_x)
            gdf["position_y"] = gdf.BBOX.apply(lambda bbox: round(bbox.min_y + translate_y) // scale_y)

        return grid

    def _check_target_compatibility(self, target_area_manager: AreaManager) -> None:
        """Checks if source and target managers are compatible to define a transformation between them."""
        if not isinstance(target_area_manager, UtmZoneAreaManager):
            raise NotImplementedError(
                f"Grid transformation is only supported into another {self.__class__.__name__}, but got "
                f"{target_area_manager.__class__.__name__}"
            )

        for param in ["offset_x", "offset_y"]:
            if getattr(self.config, param) != getattr(target_area_manager.config, param):
                raise NotImplementedError(
                    "Cannot transform grid because source and target area managers have different values of config "
                    f"parameter {param}."
                )
