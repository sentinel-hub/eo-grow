"""
For working with pre-defined grids
"""
import logging
from typing import Any, List

import fs
import geopandas as gpd
import shapely.ops
from pydantic import Field

from sentinelhub import CRS, Geometry

from ...utils.types import Path
from ...utils.vector import concat_gdf
from ..schemas import ManagerSchema
from .base import AreaManager

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
    config: Schema  # type: ignore

    def get_area_dataframe(self, *, crs: CRS = CRS.WGS84, **_: Any) -> gpd.GeoDataFrame:
        """Provides a single dataframe that defines an AOI

        :param crs: A CRS of the dataframe
        """
        grid = self.get_grid()
        return concat_gdf(grid, reproject_crs=crs)

    def get_area_geometry(self, *, crs: CRS = CRS.WGS84, **_: Any) -> Geometry:
        """Provides AOI geometry by joining grid geometries

        :param crs: A CRS in which grid geometries will be joined
        """
        area_filename = self._construct_file_path(prefix="area")

        if self.storage.filesystem.exists(area_filename):
            return self._load_area_geometry(area_filename)

        area_df = self.get_area_dataframe(crs=crs)

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

    @staticmethod
    def has_region_filter(*_: Any, **__: Any) -> bool:
        """Doesn't support filtering by a region"""
        return False

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
