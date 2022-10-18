"""
Defines pipelines for ingesting and modifying BYOC collections.
"""
import datetime
import datetime as dt
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, cast

import fs
import geopandas as gpd
import rasterio
from pydantic import Field, validator

from sentinelhub import CRS, BBox, DataCollection, SentinelHubBYOC
from sentinelhub.api.byoc import ByocCollection, ByocTile
from sentinelhub.geometry import Geometry

from eogrow.core.config import RawConfig

from ..core.pipeline import Pipeline
from ..utils.types import JsonDict
from ..utils.validators import (
    ensure_defined_together,
    ensure_exactly_one_defined,
    optional_field_validator,
    parse_data_collection,
)
from .export_maps import TIMESTAMP_FORMAT

LOGGER = logging.getLogger(__name__)


class IngestByocTilesPipeline(Pipeline):
    """Ingests .tiff files to a BYOC collection. Based on `ExportMapsPipeline` output for timeless features."""

    class Schema(Pipeline.Schema):
        byoc_tile_folder_key: str
        file_glob_pattern: str = Field("**/*.tiff", description="Pattern used to obtain the TIFF files to use")

        new_collection_name: Optional[str] = Field(description="Used for defining a new BYOC collection.")
        existing_collection: Optional[DataCollection] = Field(description="Used when updating and reingesting.")
        _parse_byoc_collection = optional_field_validator("existing_collection", parse_data_collection, pre=True)
        _ensure_exclusion = ensure_exactly_one_defined("new_collection_name", "existing_collection")

        is_temporal: bool = Field(
            False,
            description=(
                "If the BYOC is marked as temporal the pipeline will assume that the direct parent folder of a TIFF"
                " contains the sensing time, i.e. filesystem structure follows that used by `ExportMapsPipeline`."
                " Example of such a path is `UTM_32638/2019-01-04T07-48-37/BANDS_S2_L1C.tiff`."
            ),
        )
        sensing_time: Optional[datetime.datetime] = Field(
            None, description="Sensing time (ISO format) added to BYOC tiles. Only used for timeless collections."
        )

        @validator("sensing_time", pre=True)
        def _parse_sensing_time(cls, value: Optional[str], values: Dict[str, object]) -> Optional[datetime.datetime]:
            is_temporal = values["is_temporal"]
            if is_temporal and value is None:
                return None
            if not is_temporal and value is not None:
                return datetime.datetime.fromisoformat(value)
            raise AssertionError("Sensing time should be set for timeless BYOC collections.")

        cover_geometry_folder_key: Optional[str] = Field(description="Folder for supplying a custom cover geometry.")
        cover_geometry: Optional[str] = Field(description="Specifies a geometry file describing the cover geometry.")
        _ensure_cover_geometry = ensure_defined_together("cover_geometry_folder_key", "cover_geometry")

        reingest_existing: bool = Field(False, description="Whether to reingest or skip already ingested tiles.")

    config: Schema

    def __init__(self, config: Schema, raw_config: Optional[RawConfig] = None):
        super().__init__(config, raw_config)
        if not self.storage.is_on_aws:
            raise ValueError("Can only ingest for projects based on S3 storage.")
        project_folder = self.storage.config.project_folder
        self.bucket_name = project_folder.replace("s3://", "").split("/")[0]
        self._cover_geometry_df: Optional[gpd.GeoDataFrame] = None

    def get_byoc_collection(self, byoc_client: SentinelHubBYOC) -> JsonDict:
        """Obtains information about the existing collection or creates a new one."""
        if self.config.new_collection_name:
            collection_spec = ByocCollection(self.config.new_collection_name, s3_bucket=self.bucket_name)
            LOGGER.info("Creating new collection.")
            response = byoc_client.create_collection(collection_spec)
            LOGGER.info("Collection %s created.", response["id"])
        else:
            existing_collection = cast(DataCollection, self.config.existing_collection)  # due to validation
            LOGGER.info("Obtaining full information for collection id %s.", existing_collection.collection_id)
            response = byoc_client.get_collection(existing_collection)
        return response

    def get_tile_paths(self) -> Dict[str, List[str]]:
        """Collects the folders and filenames of .tiff files to be ingested.

        Paths are relative to bucket, not project.
        """
        tiff_paths = self._get_tiff_paths()

        folder_to_filename_map = defaultdict(list)
        for folder, filename in map(fs.path.split, tiff_paths):
            folder_to_filename_map[folder].append(filename)

        return dict(folder_to_filename_map)

    def _get_tiff_paths(self) -> List[str]:
        """Collects the paths to .tiff files to ingest. Paths are relative to bucket, not project."""
        folder = self.storage.get_folder(key=self.config.byoc_tile_folder_key)
        filesystem = self.storage.filesystem
        return [
            self._get_byoc_compliant_path(path)
            for path, _ in filesystem.glob(path=folder, pattern=self.config.file_glob_pattern)
        ]

    def _get_byoc_compliant_path(self, relative_path: str) -> str:
        """Transforms a project relative path to a bucket-name relative path that is required by BYOC."""
        absolute_path = fs.path.combine(self.config.storage.project_folder, relative_path)  # type: ignore[attr-defined]
        return absolute_path.replace(f"s3://{self.bucket_name}/", "")  # removes s3://<bucket-name>/

    def _prepare_tile(self, folder: str, tiff_paths: List[str]) -> Optional[ByocTile]:
        """Collects all required metainfo to create a BYOC tile for the given folder."""
        some_tiff = fs.path.join(folder, tiff_paths[0])
        cover_geometry = self._get_tile_cover_geometry(some_tiff)
        if cover_geometry.geometry.is_empty:
            return None

        sensing_time = self.config.sensing_time
        if self.config.is_temporal:
            # assumes file-system structure is equal to that of ExportMapsPipeline
            timestamp_folder = fs.path.split(folder)[1]
            sensing_time = dt.datetime.strptime(timestamp_folder, TIMESTAMP_FORMAT)

        return ByocTile(folder, cover_geometry=cover_geometry, sensing_time=sensing_time)

    def _get_tile_cover_geometry(self, tiff_path: str) -> Geometry:
        """Get geometry of the tile by intersecting the tiff geometry with the general cover geometry."""
        full_path = f"s3://{self.bucket_name}/" + tiff_path
        with rasterio.open(full_path) as tiff_data:
            tiff_bounds = tiff_data.bounds
            tiff_crs = CRS(tiff_data.crs.to_epsg())
        tiff_poly = BBox([tiff_bounds.left, tiff_bounds.bottom, tiff_bounds.right, tiff_bounds.top], tiff_crs).geometry

        cover_poly = self._get_cover_geometry(tiff_crs)
        final_poly = tiff_poly.intersection(cover_poly) if cover_poly else tiff_poly
        return Geometry(final_poly, crs=tiff_crs)

    def _get_cover_geometry(self, crs: CRS) -> Optional[Geometry]:
        """Lazy-loads the cover geometry of whole area, reprojecting (and combining) in desired CRS on call."""
        if self.config.cover_geometry is None or self.config.cover_geometry_folder_key is None:
            return None

        if self._cover_geometry_df is None:
            folder_path = self.storage.get_folder(self.config.cover_geometry_folder_key)
            file_path = fs.path.join(folder_path, self.config.cover_geometry)
            with self.storage.filesystem.openbin(file_path, "r") as file_handle:
                self._cover_geometry_df = gpd.read_file(file_handle)

        return self._cover_geometry_df.to_crs(crs.pyproj_crs()).unary_union

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Runs the procedure.

        1. Creates or loads the collection,
        2. Checks existing tiles,
        3. Creates new tiles and skips/reingests existing ones.
        """
        byoc_client = SentinelHubBYOC(config=self.storage.sh_config)
        byoc_collection = self.get_byoc_collection(byoc_client)

        existing_tiles = {fs.path.dirname(tile["path"]): tile["id"] for tile in byoc_client.iter_tiles(byoc_collection)}

        for tile_folder, tiff_paths in self.get_tile_paths().items():
            if tile_folder in existing_tiles:
                # reingest or skip existing
                if self.config.reingest_existing:
                    response = byoc_client.reingest_tile(byoc_collection, existing_tiles[tile_folder])
                    LOGGER.info(f"Reingested tile for {tile_folder}{f'with response: {response}' if response else ''}.")
                else:
                    LOGGER.info(f"Tile {tile_folder} already exists, skipping.")
            else:
                tile = self._prepare_tile(tile_folder, tiff_paths)
                if tile is None:
                    LOGGER.info(f"Intersection of tile {tile_folder} with cover geometry is empty, skipping.")
                    continue
                response = byoc_client.create_tile(byoc_collection, tile)
                if "errors" in response:
                    LOGGER.info(f"Creation of tile {tile_folder} failed with response: {response}.")
                LOGGER.info(f"Created tile for {tile_folder}.")

        return [], []
