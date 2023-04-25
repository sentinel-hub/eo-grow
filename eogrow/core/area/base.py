"""Implementation of the base AreaManager."""
import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import Dict, Literal, Optional

import fiona
import fs
import geopandas as gpd
import shapely.ops
from pydantic import Field

from eolearn.core.exceptions import EODeprecationWarning
from sentinelhub import CRS, BBox, Geometry

from ...types import PatchList, RawSchemaDict
from ...utils.eopatch_list import load_names
from ...utils.fs import LocalFile
from ..base import EOGrowObject
from ..schemas import BaseSchema, ManagerSchema
from ..storage import StorageManager

LOGGER = logging.getLogger(__name__)


class AreaSchema(BaseSchema):
    filename: str
    buffer: Optional[float] = Field(
        description="Buffer that will be applied to AOI geometry. Buffer has to be in the same units as AOI CRS.",
    )
    simplification_factor: Optional[float] = Field(
        description="Tolerance factor (in CRS units) for simplifying the buffered area geometry before splitting it.",
    )


def area_schema_deprecation(cls: type, value: Optional[str], values: RawSchemaDict) -> str:
    """Warns and reconfigures when `area` is used instead of `geometry_filename`."""
    if values.get("area") is not None:
        warnings.warn(
            (
                "Use `geometry_filename` to provide the file (e.g., geojson, gpkg) with AoI. The `area` parameter is"
                " deprecated, and extra parameters like `buffer` and `simplification_factor` are no longer available;"
                " users should prepare the AoI geometry by themselves."
            ),
            EODeprecationWarning,
            stacklevel=2,
        )
        return values["area"].filename
    assert value is not None, "Specify the `geometry_filename` parameter."
    return value


class PatchListSchema(BaseSchema):
    input_folder_key: str = Field(description="The storage manager key pointing to the folder containing the file.")
    filename: str = Field(description="A JSON file containing a list of EOPatch names.", regex=r"^.+\.(json|JSON)$")


class BaseAreaManager(EOGrowObject, metaclass=ABCMeta):
    """A manager for the AOI and how it is split into EOPatches"""

    NAME_COLUMN = "eopatch_name"

    class Schema(ManagerSchema):
        patch_names: Optional[PatchListSchema] = Field(
            description="Names of EOPatches to keep when filtering in the `get_grid` method."
        )

    config: Schema

    def __init__(self, config: Schema, storage: StorageManager):
        """
        :param config: The configuration schema
        :param storage: An instance of StorageManager class
        """
        super().__init__(config)

        self.storage = storage

    @abstractmethod
    def get_area_geometry(self, *, crs: CRS = CRS.WGS84) -> Geometry:
        """Provides a dissolved geometry object of the entire AOI"""

    def get_grid(self, filtered: bool = True) -> Dict[CRS, gpd.GeoDataFrame]:
        """Provides a grid of bounding boxes which divide the AOI. Uses caching to avoid recalculations.

        The grid is split into different CRS zones. The `bounds` properties of the geometries are taken as BBox
        definitions. EOPatch names are stored in a column with identifier `self.NAME_COLUMN`.

        :return: A dictionary of GeoDataFrames that defines how the area is split into EOPatches.
        """
        grid_path = fs.path.combine(self.storage.get_cache_folder(), self.get_grid_cache_filename())

        if self.storage.filesystem.exists(grid_path):
            grid = self._load_grid(grid_path)
        else:
            grid = self._create_grid()
            self._save_grid(grid, grid_path)

        if filtered and self.config.patch_names is not None:
            folder_path = self.storage.get_folder(self.config.patch_names.input_folder_key)
            patch_names_path = fs.path.join(folder_path, self.config.patch_names.filename)
            relevant_patch_names = set(load_names(self.storage.filesystem, patch_names_path))

            for crs, geoms in grid.items():
                grid[crs] = geoms[geoms[self.NAME_COLUMN].isin(relevant_patch_names)]

            grid = {crs: geoms for crs, geoms in grid.items() if not geoms.empty}

            num_geoms = sum(map(len, grid.values()))
            if len(relevant_patch_names) != num_geoms:
                raise ValueError(
                    f"Filtration done with {len(relevant_patch_names)} unique names, but {num_geoms} patches were found"
                )

        return grid

    @abstractmethod
    def _create_grid(self) -> Dict[CRS, gpd.GeoDataFrame]:
        """Defines a new grid, which encodes how the area is split into EOPatches.

        The grid is split into different CRS zones. The `bounds` properties of the geometries are taken as BBox
        definitions. EOPatch names are stored in a column with identifier `self.NAME_COLUMN`.
        """

    def _load_grid(self, grid_path: str) -> Dict[CRS, gpd.GeoDataFrame]:
        """A method that loads the bounding box grid from the cache folder."""
        LOGGER.info("Loading grid from %s", grid_path)

        grid = {}
        with LocalFile(grid_path, mode="r", filesystem=self.storage.filesystem) as local_file:
            for crs_layer in fiona.listlayers(local_file.path):
                data = gpd.read_file(local_file.path, layer=crs_layer, engine=self.storage.config.geopandas_backend)
                grid[CRS(data.crs)] = data

        return grid

    def _save_grid(self, grid: Dict[CRS, gpd.GeoDataFrame], grid_path: str) -> None:
        """A method that saves the bounding box grid to the cache folder."""
        LOGGER.info("Saving grid to %s", grid_path)

        with LocalFile(grid_path, mode="w", filesystem=self.storage.filesystem) as local_file:
            for _, crs_grid in grid.items():
                crs_grid.to_file(
                    local_file.path,
                    driver="GPKG",
                    encoding="utf-8",
                    layer=f"Grid EPSG:{crs_grid.crs.to_epsg()}",
                    engine=self.storage.config.geopandas_backend,
                )

    @abstractmethod
    def get_grid_cache_filename(self) -> str:
        """Provides a filename that is used for caching the grid, including the file extensions (likely .gpkg).

        Should ensure that two different grids don't clash.
        """

    def get_patch_list(self) -> PatchList:
        """Returns a list of eopatch names and appropriate BBoxes."""

        named_bboxes = []
        for crs, grid in self.get_grid(filtered=True).items():
            for _, row in grid.iterrows():
                named_bboxes.append((row.eopatch_name, BBox(row.geometry.bounds, crs=crs)))

        return named_bboxes


def get_geometry_from_file(
    filesystem: fs.base.FS, file_path: str, geopandas_engine: Literal["fiona", "pyogrio"] = "fiona"
) -> Geometry:
    """Provides a single geometry object of entire AOI"""
    with LocalFile(file_path, mode="r", filesystem=filesystem) as local_file:
        area_df = gpd.read_file(local_file.path, engine=geopandas_engine)
        area_shape = shapely.ops.unary_union(area_df.geometry)
        return Geometry(area_shape, CRS(area_df.crs))
