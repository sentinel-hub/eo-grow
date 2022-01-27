"""
Area managers working with UTM CRS
"""
import logging
from typing import List

import geopandas as gpd
from pydantic import Field

from sentinelhub import Geometry, UtmZoneSplitter

from .base import AreaManager

LOGGER = logging.getLogger(__name__)


class UtmZoneAreaManager(AreaManager):
    """Area manager that splits grid per UTM zones"""

    class Schema(AreaManager.Schema):
        patch_size_x: int = Field(description="A width of each EOPatch in meters")
        patch_size_y: int = Field(description="A height of each EOPatch in meters")
        patch_buffer: float = 0
        offset_x: float = Field(0, description="An offset of tiling grid in horizontal dimension")
        offset_y: float = Field(0, description="An offset of tiling grid in vertical dimension")

    def _get_grid_filename_params(self) -> List[object]:
        """A list of parameters specific to UTM zone splitting"""
        return [
            self.config.patch_size_x,
            self.config.patch_size_y,
            self.config.patch_buffer,
            self.config.offset_x,
            self.config.offset_y,
        ]

    def _create_new_split(self, area_geometry: Geometry) -> List[gpd.GeoDataFrame]:
        """Uses UtmZoneSplitter to create a grid"""
        LOGGER.info("Splitting area geometry into UTM zone grid")
        splitter = UtmZoneSplitter(
            [area_geometry.geometry],
            crs=area_geometry.crs,
            bbox_size=(self.config.patch_size_x, self.config.patch_size_y),
            offset=(self.config.offset_x, self.config.offset_y),
        )
        bbox_list = splitter.get_bbox_list(buffer=self.config.patch_buffer)
        info_list = splitter.get_info_list()

        for tile_info in info_list:
            tile_info["index_n"] = tile_info["index"]

        return self._to_dataframe_grid(bbox_list, info_list, ["index_n", "index_x", "index_y"])
