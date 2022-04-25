"""
A base AreaManager implementation
"""
import logging
from collections import defaultdict
from typing import Any, List, Optional

import fiona
import fs
import geopandas as gpd
import shapely.ops
from pydantic import Field

from sentinelhub import CRS, BBox, Geometry

from ...utils.fs import LocalFile
from ...utils.grid import GridTransformation
from ...utils.types import Path
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
        region_column_name: Optional[str] = Field(description="Name of the column in area file which contains regions.")
        region_names: Optional[List[str]] = Field(
            description="A list of regions which will be used. By default all regions are used."
        )

    config: Schema

    def __init__(self, config: Schema, storage: StorageManager):
        """
        :param config: A configuration file
        :param storage: An instance of StorageManager class
        """
        super().__init__(config)

        self.storage = storage

    def get_area_dataframe(self, *, ignore_region_filter: bool = False) -> gpd.GeoDataFrame:
        """Provides a GeoDataFrame with the AOI, which can be split over multiple sub-areas, each in a separate row.

        Current implementation is intentionally working without a LocalFile abstraction in order to be able to read
        formats consisting of multiple files (e.g. Shapefile or Geodatabase)

        :param ignore_region_filter: If `True` it will not filter by given region config parameters
        :return: A GeoDataFrame containing the unmodified area shape
        """
        filename = fs.path.join(self.storage.get_input_data_folder(), self.config.area_filename)
        with self.storage.filesystem.openbin(filename, "r") as file_handle:
            area_df = gpd.read_file(file_handle)

        if self.has_region_filter() and not ignore_region_filter:
            area_df = area_df[area_df[self.config.region_column_name].isin(self.config.region_names)]

        return area_df

    def get_area_geometry(self, *, ignore_region_filter: bool = False) -> Geometry:
        """Provides a single geometry object of entire AOI

        :param ignore_region_filter: If `True` it will not filter by given region config parameters
        """
        area_filename = self._construct_file_path(prefix="area", ignore_region_filter=ignore_region_filter)

        if self.storage.filesystem.exists(area_filename):
            return self._load_area_geometry(area_filename)

        area_df = self.get_area_dataframe(ignore_region_filter=ignore_region_filter)
        area_geometry = self._process_area_geometry(area_df)

        self._save_area_geometry(area_geometry, area_filename)
        return area_geometry

    def get_grid(self, *, add_bbox_column: bool = False, ignore_region_filter: bool = False) -> List[gpd.GeoDataFrame]:
        """Provides a grid of bounding boxes which divide the AOI

        :param add_bbox_column: If `True` a new column with BBox objects will be added.
        :param ignore_region_filter: If `True` it will not filter by given region config parameters.
        :return: A list of GeoDataFrames containing bounding box geometries and their info. Bounding boxes are divided
            into GeoDataFrames per CRS.
        """
        grid_filename = self._construct_file_path(prefix="grid", ignore_region_filter=ignore_region_filter)

        if self.storage.filesystem.exists(grid_filename):
            grid = self._load_grid(grid_filename)
        else:
            grid = self._create_and_save_grid(grid_filename, ignore_region_filter)

        if add_bbox_column:
            self._add_bbox_column(grid)

        return grid

    def cache_grid(self, grid: Optional[List[gpd.GeoDataFrame]] = None, ignore_region_filter: bool = False) -> None:
        """Checks if grid is cached in the storage. If it is not, it will create and cache it.

        :param grid: If provided, this grid will be cached. Otherwise, it will generate a new grid.
        :param ignore_region_filter: If `True` it will not filter by given region config parameters.
        """
        grid_filename = self._construct_file_path(prefix="grid", ignore_region_filter=ignore_region_filter)

        if not self.storage.filesystem.exists(grid_filename):
            if grid is None:
                self._create_and_save_grid(grid_filename, ignore_region_filter)
            else:
                self._save_grid(grid, grid_filename)

    def get_grid_size(self, **kwargs: Any) -> int:
        """Calculates the number of elements of the grid

        :param kwargs: Parameters that are propagated to `get_grid` method
        :return: The number of bounding boxes in the grid
        """
        grid = self.get_grid(**kwargs)
        return sum([len(df.index) for df in grid])

    def transform_grid(self, target_area_manager: "AreaManager") -> List[GridTransformation]:
        """This method is used to define how a grid, defined by this area manager, will be transformed into a grid,
        defined by given target area manager. It calculates transformation objects between groups of bounding boxes
        from source grid and groups of bounding boxes from target grid.

        Every area manager should implement its own process of transformation and define which target area managers it
        supports."""
        raise NotImplementedError

    def has_region_filter(self) -> bool:
        """Checks region filter is set in the configuration"""
        return self.config.region_column_name is not None and self.config.region_names is not None

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
            area_gdf = gpd.read_file(local_file.path)

        return Geometry(area_gdf.geometry[0], CRS(area_gdf.crs))

    def _save_area_geometry(self, area_geometry: Geometry, filename: str) -> None:
        """Saves processed geometry of an area"""
        LOGGER.info("Saving area geometry to %s", filename)

        area_gdf = gpd.GeoDataFrame(geometry=[area_geometry.geometry], crs=area_geometry.crs.pyproj_crs())

        with LocalFile(filename, mode="w", filesystem=self.storage.filesystem) as local_file:
            area_gdf.to_file(local_file.path, driver="GPKG", encoding="utf-8")

    def _create_and_save_grid(self, grid_filename: str, ignore_region_filter: bool) -> List[gpd.GeoDataFrame]:
        """Defines a new grid and saves it."""
        if self.has_region_filter() and not ignore_region_filter:
            full_grid = self.get_grid(ignore_region_filter=True)
            filtered_area_geometry = self.get_area_geometry(ignore_region_filter=False)

            grid = self._filter_grid(full_grid, filtered_area_geometry)
        else:
            area_geometry = self.get_area_geometry(ignore_region_filter=ignore_region_filter)
            grid = self._create_new_split(area_geometry)

        self._save_grid(grid, grid_filename)
        return grid

    def _load_grid(self, filename: str) -> List[gpd.GeoDataFrame]:
        """A method that loads bounding box grid saved in a cache folder"""
        LOGGER.info("Loading grid from %s", filename)

        grid = []
        with LocalFile(filename, mode="r", filesystem=self.storage.filesystem) as local_file:
            for crs_layer in fiona.listlayers(local_file.path):
                grid.append(gpd.read_file(local_file.path, layer=crs_layer))

        return grid

    def _save_grid(self, grid: List[gpd.GeoDataFrame], filename: str) -> None:
        """A method that saves bounding box grid in a cache folder"""
        LOGGER.info("Saving grid to %s", filename)

        with LocalFile(filename, mode="w", filesystem=self.storage.filesystem) as local_file:
            for crs_grid in grid:
                crs_grid.to_file(
                    local_file.path, driver="GPKG", encoding="utf-8", layer=f"Grid EPSG:{crs_grid.crs.to_epsg()}"
                )

    def _construct_file_path(self, *, prefix: str, suffix: str = "gpkg", ignore_region_filter: bool = False) -> str:
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

        if self.has_region_filter() and not ignore_region_filter:
            filename_params.append(self.config.region_column_name)
            filename_params.extend(self.config.region_names or [])

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
    def _filter_grid(grid: List[gpd.GeoDataFrame], geometry: Geometry) -> List[gpd.GeoDataFrame]:
        """Filters grid by keeping only tiles that intersect with the given geometry"""
        filtered_grid = []
        for dataframe in grid:
            projected_shape = geometry.transform(dataframe.crs).geometry
            filtered_dataframe = dataframe[dataframe.geometry.intersects(projected_shape)].copy()

            if len(filtered_dataframe.index) > 0:
                filtered_grid.append(filtered_dataframe)

        return filtered_grid

    @staticmethod
    def _add_bbox_column(grid: List[gpd.GeoDataFrame]) -> None:
        """Adds a column with bounding boxes to all dataframes in a grid"""
        for bbox_df in grid:
            crs = CRS(bbox_df.crs)
            bbox_df["BBOX"] = bbox_df.geometry.apply(lambda geo: BBox(geo.bounds, crs))
