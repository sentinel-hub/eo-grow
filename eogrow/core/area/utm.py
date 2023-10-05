"""Area manager implementation for automated UTM CRS grids."""

from __future__ import annotations

import logging
from collections import defaultdict

import fs
import geopandas as gpd
from geopandas import GeoDataFrame
from pydantic import Field

from sentinelhub import CRS, Geometry, UtmZoneSplitter

from ..schemas import BaseSchema
from .base import BaseAreaManager, get_geometry_from_file

LOGGER = logging.getLogger(__name__)


class PatchSchema(BaseSchema):
    size_x: int = Field(description="A width of each EOPatch in meters")
    size_y: int = Field(description="A height of each EOPatch in meters")
    buffer_x: float = Field(0, description="Number of meters by which to increase the tile size to left and right.")
    buffer_y: float = Field(0, description="Number of meters by which to increase the tile size to up and down.")


class UtmZoneAreaManager(BaseAreaManager):
    """Area manager that splits the area per UTM zone"""

    class Schema(BaseAreaManager.Schema):
        geometry_filename: str = Field(
            description="Name of the file that defines the AoI geometry, located in the input data folder."
        )
        patch: PatchSchema

        offset_x: float = Field(0, description="An offset of tiling grid in horizontal dimension")
        offset_y: float = Field(0, description="An offset of tiling grid in vertical dimension")

    config: Schema

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84) -> Geometry:
        file_path = fs.path.join(self.storage.get_input_data_folder(), self.config.geometry_filename)
        return get_geometry_from_file(
            filesystem=self.storage.filesystem,
            file_path=file_path,
            geopandas_engine=self.storage.config.geopandas_backend,
        ).transform(crs)

    def _create_grid(self) -> dict[CRS, GeoDataFrame]:
        """Uses UtmZoneSplitter to create a grid"""
        area_geometry = self.get_area_geometry()
        LOGGER.info("Splitting area geometry into UTM zone grid")
        splitter = UtmZoneSplitter(
            [area_geometry.geometry],
            crs=area_geometry.crs,
            bbox_size=(self.config.patch.size_x, self.config.patch.size_y),
            offset=(self.config.offset_x, self.config.offset_y),
        )

        bbox_list, info_list = splitter.get_bbox_list(), splitter.get_info_list()

        absolute_buffer = self.config.patch.buffer_x, self.config.patch.buffer_y
        if absolute_buffer != (0, 0):
            bbox_list = [bbox.buffer(absolute_buffer, relative=False) for bbox in bbox_list]

        crs_to_patches = defaultdict(list)
        zfill_length = len(str(len(bbox_list) - 1))
        for i, (bbox, info) in enumerate(zip(bbox_list, info_list)):
            i_x, i_y = info["index_x"], info["index_y"]
            name = f"eopatch-id-{i:0{zfill_length}}-col-{i_x}-row-{i_y}"
            crs_to_patches[bbox.crs].append((name, bbox.geometry))

        grid = {}
        for crs, named_bbox_geoms in crs_to_patches.items():
            names, geoms = zip(*named_bbox_geoms)
            grid[crs] = gpd.GeoDataFrame({self.NAME_COLUMN: names}, geometry=list(geoms), crs=crs.pyproj_crs())

        return grid

    def get_grid_cache_filename(self) -> str:
        input_filename = fs.path.basename(self.config.geometry_filename)
        input_filename = input_filename.rsplit(".", 1)[0]

        raw_params = [
            input_filename,
            self.config.patch.size_x,
            self.config.patch.size_y,
            self.config.patch.buffer_x,
            self.config.patch.buffer_y,
            self.config.offset_x,
            self.config.offset_y,
        ]
        params = [str(param) for param in raw_params]

        return f"{self.__class__.__name__}_{'_'.join(params)}.gpkg"
