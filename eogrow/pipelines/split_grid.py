"""Implements a pipeline that creates a finer grid and splits EOPatches accordingly."""

from __future__ import annotations

import itertools as it
import logging
from collections import defaultdict
from typing import List, Literal, Tuple, Union

import fs
import geopandas as gpd
from pydantic import Field

from eolearn.core import EONode, EOWorkflow, LoadTask, OverwritePermission
from eolearn.core.types import Feature
from sentinelhub import CRS, BBox
from sentinelhub.geometry import Geometry

from ..core.area.batch import BatchAreaManager
from ..core.area.utm import UtmZoneAreaManager
from ..core.pipeline import Pipeline
from ..tasks.common import SkippableSaveTask
from ..tasks.spatial import SpatialSliceTask
from ..types import ExecKwargs
from ..utils.fs import LocalFile
from ..utils.grid import split_bbox
from ..utils.validators import ensure_storage_key_presence

LOGGER = logging.getLogger(__name__)

NamedBBox = Tuple[str, BBox]


class SplitGridPipeline(Pipeline):
    """Pipeline that creates a finer grid and splits EOPatches accordingly.

    The new grid is output as a geopackage file, which can be used with a `CustomAreaManager`.
    The name of the column with eopatch names is `eopatch_name`.
    """

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="A storage manager key pointing to the folder where the data will be loaded from."
        )
        _ensure_input_folder_key = ensure_storage_key_presence("input_folder_key")
        eopatch_output_folder_key: str = Field(
            description="A storage manager key pointing to the folder where the data will be saved."
        )
        _ensure_eopatch_output_folder_key = ensure_storage_key_presence("eopatch_output_folder_key")

        grid_output_folder_key: str = Field(
            description="A storage manager key of where to save the resulting split grid."
        )
        _ensure_grid_output_folder_key = ensure_storage_key_presence("grid_output_folder_key")

        subsplit_grid_filename: str = Field(
            description="Filename of new grid, which can be used in `CustomAreaManager`.", regex=r"^.+\.gpkg$"
        )
        features: List[Feature] = Field(description="Which features will be loaded and adapted to new grid.")
        raise_misaligned: bool = Field(
            True,
            description="Raise an error if spatially splitting the EOPatch produces misalignments.",
        )

        split_x: int = Field(2, description="Number of splits done on each EOPatch along the x-axis.")
        split_y: int = Field(2, description="Number of splits done on each EOPatch along the y-axis.")

        buffer: Union[Literal["auto"], Tuple[float, float]] = Field(
            "auto",
            description=(
                "Amount of buffer present on existing EOPatches, provided as (buffer_x, buffer_y) in CRS units. Applies"
                " same buffer to split EOPatches. The pipeline tries to obtain this information from area manager, but"
                " not all area manager classes are supported."
            ),
        )

        prune: bool = Field(
            True, description="Remove all split EOPatches that don't intersect with the area manager AOI."
        )

        skip_existing: Literal[False] = False

    config: Schema

    def run_procedure(self) -> tuple[list[str], list[str]]:
        buffer_x, buffer_y = self._get_buffer()

        patch_list = self.get_patch_list()
        area = self.area_manager.get_area_geometry()
        area_projection_cache = {area.crs: area}

        bbox_splits = []
        for named_bbox in patch_list:
            split_bboxes = split_bbox(
                named_bbox,
                split_x=self.config.split_x,
                split_y=self.config.split_y,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
            )

            if self.config.prune:
                split_bboxes = self._keep_intersecting(area, area_projection_cache, split_bboxes)

            bbox_splits.append((named_bbox, split_bboxes))

        self.save_new_grid(bbox_splits)

        workflow = self.build_workflow()
        patch_list = self.get_patch_list()
        exec_args = self.get_execution_arguments(workflow, bbox_splits)

        finished, failed, _ = self.run_execution(workflow, exec_args)

        return finished, failed

    def _keep_intersecting(
        self, area: Geometry, area_cache: dict[CRS, Geometry], split_bboxes: list[NamedBBox]
    ) -> list[NamedBBox]:
        """Assumes all bboxes in a split share the same CRS. Only keeps the ones that intersect the AOI."""
        if not split_bboxes:
            return []
        _, some_bbox = split_bboxes[0]
        crs = some_bbox.crs

        if crs not in area_cache:
            area_cache[crs] = area.transform(crs)

        return [bbox for bbox in split_bboxes if _intersects_area(bbox, area_cache[crs])]

    def _get_buffer(self) -> tuple[float, float]:
        """Infers buffer from AreaManager schemas if possible."""
        if self.config.buffer != "auto":
            return self.config.buffer

        area_config = self.area_manager.config
        if isinstance(area_config, UtmZoneAreaManager.Schema):
            return area_config.patch.buffer_x, area_config.patch.buffer_y
        if isinstance(area_config, BatchAreaManager.Schema):
            res = area_config.resolution
            return area_config.tile_buffer_x * res, area_config.tile_buffer_y * res
        raise ValueError(
            f"Cannot infer buffer from area manager `{type(self.area_manager)}`. Please set the `buffer` parameter."
        )

    def build_workflow(self) -> EOWorkflow:
        input_path = self.storage.get_folder(self.config.input_folder_key)
        load_node = EONode(LoadTask(input_path, filesystem=self.storage.filesystem, features=self.config.features))

        processing_nodes = []
        output_path = self.storage.get_folder(self.config.eopatch_output_folder_key)
        for _ in range(self.config.split_x * self.config.split_y):
            slice_task = SpatialSliceTask(self.config.features, raise_misaligned=self.config.raise_misaligned)
            slice_node = EONode(slice_task, inputs=[load_node])

            save_task = SkippableSaveTask(
                output_path,
                filesystem=self.storage.filesystem,
                features=self.config.features,
                overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                use_zarr=self.storage.config.use_zarr,
            )
            save_node = EONode(save_task, inputs=[slice_node])
            processing_nodes.append(save_node)

        return EOWorkflow.from_endnodes(*processing_nodes)

    def get_execution_arguments(  # type: ignore[override]
        self, workflow: EOWorkflow, bbox_splits: list[tuple[NamedBBox, list[NamedBBox]]]
    ) -> ExecKwargs:
        nodes = workflow.get_nodes()
        load_node = nodes[0]
        save_nodes = [node for node in nodes if isinstance(node.task, SkippableSaveTask)]
        slice_nodes = [save_node.inputs[0] for save_node in save_nodes]

        exec_args: ExecKwargs = {}
        for (orig_name, _), split_bboxes in bbox_splits:
            patch_args: dict[EONode, dict[str, object]] = {load_node: dict(eopatch_folder=orig_name)}
            # Since some bboxes might get filtered out, the remaining slice and save nodes should get None arguments
            split_bboxes_iter = it.chain(split_bboxes, it.repeat((None, None)))

            for slice_node, save_node, (subbox_name, subbox) in zip(slice_nodes, save_nodes, split_bboxes_iter):
                patch_args[slice_node] = dict(bbox=subbox, skip=(subbox is None))
                patch_args[save_node] = dict(eopatch_folder=subbox_name)

            exec_args[orig_name] = patch_args

        return exec_args

    def save_new_grid(self, bbox_splits: list[tuple[NamedBBox, list[NamedBBox]]]) -> None:
        """Organizes BBoxes into multiple GeoDataFrames that are then saved as layers of a GPKG file."""
        crs_groups = defaultdict(list)
        for _, new_bboxes in bbox_splits:
            for name, bbox in new_bboxes:
                crs_groups[bbox.crs].append((name, bbox))

        new_grid_path = fs.path.join(
            self.storage.get_folder(self.config.grid_output_folder_key), self.config.subsplit_grid_filename
        )
        with LocalFile(new_grid_path, mode="w", filesystem=self.storage.filesystem) as local_file:
            for crs, named_bboxes in crs_groups.items():
                names = [name for name, _ in named_bboxes]
                geometries = [bbox.geometry for _, bbox in named_bboxes]

                crs_grid = gpd.GeoDataFrame({"eopatch_name": names}, geometry=geometries, crs=crs.pyproj_crs())
                crs_grid.to_file(
                    local_file.path,
                    driver="GPKG",
                    encoding="utf-8",
                    layer=f"Grid EPSG:{crs_grid.crs.to_epsg()}",
                    engine=self.storage.config.geopandas_backend,
                )


def _intersects_area(named_bbox: NamedBBox, area: Geometry) -> bool:
    _, bbox = named_bbox
    if area.crs is not bbox.crs:
        raise ValueError("CRS of area and BBox do not match, cannot calculate intersection.")
    return area.geometry.intersects(bbox.geometry)
