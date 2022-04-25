"""
Area managers working with UTM CRS
"""
import logging
from typing import Any, List

from geopandas import GeoDataFrame
from pydantic import Field

from sentinelhub import Geometry, UtmZoneSplitter

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
