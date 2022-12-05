"""Area manager implementation for custom grids."""
import logging
from typing import Any, Dict, List

import fs
import geopandas as gpd
import shapely.ops
from pydantic import Field

from sentinelhub import CRS, Geometry

from ...utils.types import Path
from ...utils.vector import concat_gdf
from ..schemas import ManagerSchema
from .base import AreaManager, BaseAreaManager

LOGGER = logging.getLogger(__name__)


class CustomGridAreaManager(AreaManager):
    """Area manager that works with a pre-defined grid of EOPatches"""

    class Schema(ManagerSchema):
        grid_filename: Path = Field(
            description=(
                "A Geopackage with a collection of bounding boxes and attributes that will define EOPatches. In case "
                "bounding boxes are in multiple CRS then each Geopackage layer should contain bounding boxes from one "
                "CRS."
            ),
        )

    # TODO: The AreaManager needs to be reworked until this is no longer an issue
    config: Schema  # type: ignore[assignment]

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84, **_: Any) -> Geometry:
        """Provides AOI geometry by joining grid geometries

        :param crs: A CRS in which grid geometries will be joined
        """
        area_filename = self._construct_file_path(prefix="area")

        if self.storage.filesystem.exists(area_filename):
            return self._load_area_geometry(area_filename)

        area_df = concat_gdf(self.get_grid(), reproject_crs=crs)

        LOGGER.info("Calculating a unary union of the geometries")
        area_shape = shapely.ops.unary_union(area_df.geometry)
        area_geometry = Geometry(area_shape, crs)

        self._save_area_geometry(area_geometry, area_filename)
        return area_geometry

    def get_grid(self, *, add_bbox_column: bool = False, **_: Any) -> List[gpd.GeoDataFrame]:
        """Provides a grid of bounding boxes which divide the AOI

        :param add_bbox_column: If `True` a new column with BBox objects will be added.
        :return: A list of GeoDataFrames containing bounding box geometries and their info. Bounding boxes are divided
            into GeoDataFrames per CRS.
        """
        grid_filename = fs.path.combine(self.storage.get_input_data_folder(), self.config.grid_filename)
        grid = self._load_grid(grid_filename)

        if add_bbox_column:
            self._add_bbox_column(grid)

        return grid

    def _construct_file_path(self, *, prefix: str, suffix: str = "gpkg", **_: Any) -> str:
        """Provides a file path of a cached file"""
        input_filename = self.config.grid_filename.rsplit(".", 1)[0]
        input_filename = fs.path.basename(input_filename)

        filename_params = [
            prefix,
            input_filename,
            self.__class__.__name__,
        ]

        filename = "_".join(map(str, filename_params))
        filename = filename.replace(" ", "_")

        return fs.path.combine(self.storage.get_cache_folder(), f"{filename}.{suffix}")


class NewCustomGridAreaManager(BaseAreaManager):
    """Area manager that works with a pre-defined grid of EOPatches"""

    class Schema(BaseAreaManager.Schema):
        grid_folder_key: str = Field("input_data", description="Which folder the grid file is in.")
        grid_filename: Path = Field(
            description=(
                "A Geopackage with a collection of bounding boxes and attributes that will define EOPatches. In"
                " case bounding boxes are in multiple CRS then each Geopackage layer should contain bounding boxes"
                " from one CRS."
            ),
            regex=r"^.+\..+$",
        )
        name_column: str = Field(description="Name of the column containing EOPatch names.")

    config: Schema

    def _create_grid(self) -> Dict[CRS, gpd.GeoDataFrame]:
        grid_path = fs.path.combine(self.storage.get_folder(self.config.grid_folder_key), self.config.grid_filename)
        grid = self._load_grid(grid_path)

        for crs, crs_gdf in grid.items():
            # Correct name of eoptach-name-column, drop all non-significant ones
            names = crs_gdf[self.config.name_column]
            grid[crs] = gpd.GeoDataFrame(geometry=crs_gdf.geometry, data={self.NAME_COLUMN: names}, crs=crs_gdf.crs)

        return grid

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84) -> Geometry:
        all_grid_gdfs = list(self.get_grid().values())
        area_df = concat_gdf(all_grid_gdfs, reproject_crs=crs)

        LOGGER.info("Calculating a unary union of the area geometries")
        area_shape = shapely.ops.unary_union(area_df.geometry)
        return Geometry(area_shape, crs)

    def get_grid_cache_filename(self) -> str:
        input_filename = fs.path.basename(self.config.grid_filename)
        input_filename = input_filename.rsplit(".", 1)[0]

        return f"{self.__class__.__name__}_{input_filename}.gpkg"
