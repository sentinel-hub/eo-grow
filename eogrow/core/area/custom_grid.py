"""Area manager implementation for custom grids."""

from __future__ import annotations

import logging

import fs
import geopandas as gpd
import shapely.ops
from pydantic import Field

from sentinelhub import CRS, Geometry

from ...utils.vector import concat_gdf
from .base import BaseAreaManager

LOGGER = logging.getLogger(__name__)


class CustomGridAreaManager(BaseAreaManager):
    """Area manager that works with a pre-defined grid of EOPatches"""

    class Schema(BaseAreaManager.Schema):
        grid_folder_key: str = Field("input_data", description="Storage key pointing to the folder with the grid file.")
        grid_filename: str = Field(
            description=(
                "A Geopackage with a collection of bounding boxes and attributes that define EOPatches. If bounding"
                " boxes are in multiple CRS then each Geopackage layer should contain bounding boxes from one CRS."
            ),
            regex=r"^.+\..+$",
        )
        name_column: str = Field(description="Name of the column containing EOPatch names.")

    config: Schema

    def _create_grid(self) -> dict[CRS, gpd.GeoDataFrame]:
        grid_path = fs.path.combine(self.storage.get_folder(self.config.grid_folder_key), self.config.grid_filename)
        grid = self._load_grid(grid_path)

        for crs, crs_gdf in grid.items():
            # Correct name of eopatch-name-column, drop all non-significant ones
            names = crs_gdf[self.config.name_column]
            grid[crs] = gpd.GeoDataFrame(geometry=crs_gdf.geometry, data={self.NAME_COLUMN: names}, crs=crs_gdf.crs)

        return grid

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84) -> Geometry:
        all_grid_gdfs = list(self.get_grid(filtered=True).values())
        area_df = concat_gdf(all_grid_gdfs, reproject_crs=crs)

        LOGGER.info("Calculating a unary union of the area geometries")
        area_shape = shapely.ops.unary_union(area_df.geometry)
        return Geometry(area_shape, crs)

    def get_grid_cache_filename(self) -> str:
        input_filename = fs.path.basename(self.config.grid_filename)
        input_filename = input_filename.rsplit(".", 1)[0]

        return f"{self.__class__.__name__}_{input_filename}.gpkg"
