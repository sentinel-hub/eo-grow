"""Implementation of the base AreaManager."""
import logging
from collections import defaultdict
from typing import List, Optional

import fiona
import fs
import geopandas as gpd
import shapely.ops
from pydantic import Field

from sentinelhub import CRS, BBox, Geometry

from ...types import Path
from ...utils.fs import LocalFile
from ...utils.grid import GridTransformation
from ...utils.vector import count_points
from ..base import EOGrowObject
from ..schemas import ManagerSchema
from ..storage import StorageManager

LOGGER = logging.getLogger(__name__)


class AreaManager(EOGrowObject):
    """A class that manages AOI and how it is split into EOPatches"""

    class Schema(ManagerSchema):
        area_filename: Path
        area_buffer: Optional[float] = Field(
            description=(
                "Buffer that will be applied to AOI geometry. Buffer has to be in the same units as AOI CRS. "
                "In case buffer is too small, relatively to AOI size, it won't have any effect."
            ),
        )
        area_simplification_factor: Optional[float] = Field(
            description=(
                "A tolerance factor in CRS units how much the buffered area geometry will be simplified before "
                "splitting."
            ),
        )

    config: Schema

    def __init__(self, config: Schema, storage: StorageManager):
        """
        :param config: A configuration file
        :param storage: An instance of StorageManager class
        """
        super().__init__(config)

        self.storage = storage

    def get_area_geometry(self) -> Geometry:
        """Provides a single geometry object of entire AOI"""
        area_filename = self._construct_file_path(prefix="area")

        if self.storage.filesystem.exists(area_filename):
            return self._load_area_geometry(area_filename)

        aoi_filename = fs.path.join(self.storage.get_input_data_folder(), self.config.area_filename)
        with LocalFile(aoi_filename, mode="r", filesystem=self.storage.filesystem) as local_file:
            area_df = gpd.read_file(local_file.path, engine=self.storage.config.geopandas_backend)

        area_geometry = self._process_area_geometry(area_df)

        self._save_area_geometry(area_geometry, area_filename)
        return area_geometry

    def get_grid(self, *, add_bbox_column: bool = False) -> List[gpd.GeoDataFrame]:
        """Provides a grid of bounding boxes which divide the AOI

        :param add_bbox_column: If `True` a new column with BBox objects will be added.
        :return: A list of GeoDataFrames containing bounding box geometries and their info. Bounding boxes are divided
            into GeoDataFrames per CRS.
        """
        grid_filename = self._construct_file_path(prefix="grid")

        if self.storage.filesystem.exists(grid_filename):
            grid = self._load_grid(grid_filename)
        else:
            grid = self._create_and_save_grid(grid_filename)

        if add_bbox_column:
            self._add_bbox_column(grid)

        return grid

    def cache_grid(self, grid: Optional[List[gpd.GeoDataFrame]] = None) -> None:
        """Checks if grid is cached in the storage. If it is not, it will create and cache it.

        :param grid: If provided, this grid will be cached. Otherwise, it will generate a new grid.
        """
        grid_filename = self._construct_file_path(prefix="grid")

        if not self.storage.filesystem.exists(grid_filename):
            if grid is None:
                self._create_and_save_grid(grid_filename)
            else:
                self._save_grid(grid, grid_filename)

    def transform_grid(self, target_area_manager: "AreaManager") -> List[GridTransformation]:
        """This method is used to define how a grid, defined by this area manager, will be transformed into a grid,
        defined by given target area manager. It calculates transformation objects between groups of bounding boxes
        from source grid and groups of bounding boxes from target grid.

        Every area manager should implement its own process of transformation and define which target area managers it
        supports."""
        raise NotImplementedError

    def _process_area_geometry(self, area_df: gpd.GeoDataFrame) -> Geometry:
        """A method that joins area dataframe into a single geometry and applies preprocessing operations on it.

        Note: Both unary_union and buffering can take quite some time and memory when dealing with complex shapes.
        """
        LOGGER.info("Calculating a unary union of the geometries")
        area_shape = shapely.ops.unary_union(area_df.geometry)

        if self.config.area_buffer is not None:
            LOGGER.info("Applying buffer on area geometry")
            area_shape = area_shape.buffer(self.config.area_buffer)

        if self.config.area_simplification_factor is not None:
            LOGGER.info("Simplifying area geometry")
            area_shape = area_shape.simplify(self.config.area_simplification_factor, preserve_topology=True)

            point_count = count_points(area_shape)
            LOGGER.info("Simplified area shape has %d points", point_count)

        LOGGER.info("Finished processing area geometry")
        return Geometry(area_shape, CRS(area_df.crs))

    def _load_area_geometry(self, filename: str) -> Geometry:
        """Loads existing processed geometry of an area"""
        LOGGER.info("Loading cached area geometry from %s", filename)

        with LocalFile(filename, mode="r", filesystem=self.storage.filesystem) as local_file:
            area_gdf = gpd.read_file(local_file.path, engine=self.storage.config.geopandas_backend)

        return Geometry(area_gdf.geometry[0], CRS(area_gdf.crs))

    def _save_area_geometry(self, area_geometry: Geometry, filename: str) -> None:
        """Saves processed geometry of an area"""
        LOGGER.info("Saving area geometry to %s", filename)

        area_gdf = gpd.GeoDataFrame(geometry=[area_geometry.geometry], crs=area_geometry.crs.pyproj_crs())

        with LocalFile(filename, mode="w", filesystem=self.storage.filesystem) as local_file:
            area_gdf.to_file(
                local_file.path, driver="GPKG", encoding="utf-8", engine=self.storage.config.geopandas_backend
            )

    def _create_and_save_grid(self, grid_filename: str) -> List[gpd.GeoDataFrame]:
        """Defines a new grid and saves it."""
        area_geometry = self.get_area_geometry()
        grid = self._create_new_split(area_geometry)

        self._save_grid(grid, grid_filename)
        return grid

    def _load_grid(self, filename: str) -> List[gpd.GeoDataFrame]:
        """A method that loads bounding box grid saved in a cache folder"""
        LOGGER.info("Loading grid from %s", filename)

        grid = []
        with LocalFile(filename, mode="r", filesystem=self.storage.filesystem) as local_file:
            for crs_layer in fiona.listlayers(local_file.path):
                grid.append(
                    gpd.read_file(local_file.path, layer=crs_layer, engine=self.storage.config.geopandas_backend)
                )

        return grid

    def _save_grid(self, grid: List[gpd.GeoDataFrame], filename: str) -> None:
        """A method that saves bounding box grid in a cache folder"""
        LOGGER.info("Saving grid to %s", filename)

        with LocalFile(filename, mode="w", filesystem=self.storage.filesystem) as local_file:
            for crs_grid in grid:
                crs_grid.to_file(
                    local_file.path,
                    driver="GPKG",
                    encoding="utf-8",
                    layer=f"Grid EPSG:{crs_grid.crs.to_epsg()}",
                    engine=self.storage.config.geopandas_backend,
                )

    def _construct_file_path(self, *, prefix: str, suffix: str = "gpkg") -> str:
        """Provides a file path of a cached file"""
        input_filename = self.config.area_filename.rsplit(".", 1)[0]
        input_filename = fs.path.basename(input_filename)

        filename_params: List[object] = [
            prefix,
            input_filename,
            self.__class__.__name__,
            self.config.area_buffer,
            self.config.area_simplification_factor,
        ]

        if prefix == "grid":
            filename_params.extend(self._get_grid_filename_params())

        filename_params = [("" if param is None else param) for param in filename_params]
        filename = "_".join(map(str, filename_params))
        filename = filename.replace(" ", "_")

        return fs.path.combine(self.storage.get_cache_folder(), f"{filename}.{suffix}")

    def _get_grid_filename_params(self) -> List[object]:
        """Provides a list of parameters to be used for grid filename. This method has to be implemented in
        subclasses.
        """
        raise NotImplementedError

    def _create_new_split(self, area_geometry: Geometry) -> List[gpd.GeoDataFrame]:
        """Splits area geometry into a grid of bounding boxes. This method has to be implemented in subclasses."""
        raise NotImplementedError

    @staticmethod
    def _to_dataframe_grid(
        bbox_list: List[BBox], info_list: List[dict], info_columns: List[str]
    ) -> List[gpd.GeoDataFrame]:
        """Splits given lists of bounding boxes and their information by CRS and for each CRS builds a GeoDataFrame

        :param bbox_list: A list of bounding boxes
        :param info_list: A list of dictionaries with information for each bounding box
        :param info_columns: A list of keys in `info_list` dictionaries that should be used to create dataframes
        """
        crs_to_bboxes = defaultdict(list)
        for bbox, info in zip(bbox_list, info_list):
            crs_to_bboxes[bbox.crs].append((bbox.geometry, info))

        dataframe_grid = []
        for crs, bbox_prop_list in crs_to_bboxes.items():
            dataframe_grid.append(
                gpd.GeoDataFrame(
                    {column: [info[column] for _, info in bbox_prop_list] for column in info_columns},
                    geometry=[geometry for geometry, _ in bbox_prop_list],
                    crs=crs.pyproj_crs(),
                )
            )
        return dataframe_grid

    @staticmethod
    def _add_bbox_column(grid: List[gpd.GeoDataFrame]) -> None:
        """Adds a column with bounding boxes to all dataframes in a grid"""
        for bbox_df in grid:
            crs = CRS(bbox_df.crs)
            bbox_df["BBOX"] = bbox_df.geometry.apply(lambda geo: BBox(geo.bounds, crs))  # noqa B023
