"""Implements a pipeline that creates a finer grid and splits EOPatches accordingly."""
import logging
from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Union

import fs
import geopandas as gpd
from pydantic import Field

from eolearn.core import EONode, EOWorkflow, FeatureType, LoadTask, OverwritePermission, SaveTask
from sentinelhub import BBox

from ..core.area.batch import BatchAreaManager
from ..core.area.utm import UtmZoneAreaManager
from ..core.pipeline import Pipeline
from ..tasks.spatial import SpatialSliceTask
from ..utils.fs import LocalFile
from ..utils.grid import split_bbox
from ..utils.types import Feature, FeatureSpec

LOGGER = logging.getLogger(__name__)

NamedBBox = Tuple[str, BBox]


class SplitGridPipeline(Pipeline):
    """Pipeline that creates a finer grid and splits EOPatches accordingly.

    The new grid is output as a geopackage file, which can be used with a `CustomAreaManager`.
    """

    class Schema(Pipeline.Schema):
        input_folder_key: str = Field(
            description="A storage manager key pointing to the folder where the data will be loaded from."
        )
        eopatch_output_folder_key: str = Field(
            description="A storage manager key pointing to the folder where the data will be saved."
        )
        grid_output_folder_key: str = Field(
            description="A storage manager key of where to save the resulting split grid."
        )
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
        # TODO: could add AOI file?

        skip_existing: Literal[False] = False

    config: Schema

    def run_procedure(self) -> Tuple[List, List]:
        buffer_x, buffer_y = self._get_buffer()

        bboxes = self.eopatch_manager.get_bboxes(eopatch_list=self.patch_list)

        bbox_splits = []
        for named_bbox in zip(self.patch_list, bboxes):
            split_bboxes = split_bbox(
                named_bbox,
                split_x=self.config.split_x,
                split_y=self.config.split_y,
                buffer_x=buffer_x,
                buffer_y=buffer_y,
            )
            bbox_splits.append((named_bbox, split_bboxes))

        self.save_new_grid(bbox_splits)

        workflow = self.build_workflow()
        exec_args = self.get_execution_arguments(workflow, bbox_splits)

        finished, failed, _ = self.run_execution(workflow, exec_args)

        return finished, failed

    def _get_buffer(self) -> Tuple[float, float]:
        """Infers buffer from AreaManager schemas if possible."""
        if self.config.buffer != "auto":
            return self.config.buffer

        area_config = self.area_manager.config
        if isinstance(area_config, UtmZoneAreaManager.Schema):
            return area_config.patch_buffer_x, area_config.patch_buffer_y
        if isinstance(area_config, BatchAreaManager.Schema):
            res = area_config.resolution
            return area_config.tile_buffer_x * res, area_config.tile_buffer_y * res
        raise ValueError(
            f"Cannot infer buffer from area manager `{type(self.area_manager)}`. Please set the `buffer` parameter."
        )

    def build_workflow(self) -> EOWorkflow:
        features = self._get_features()

        input_path = self.storage.get_folder(self.config.input_folder_key)
        load_node = EONode(LoadTask(input_path, filesystem=self.storage.filesystem, features=features))

        processing_nodes = []
        output_path = self.storage.get_folder(self.config.eopatch_output_folder_key)
        for _ in range(self.config.split_x * self.config.split_y):
            slice_task = SpatialSliceTask(features, raise_misaligned=self.config.raise_misaligned)
            slice_node = EONode(slice_task, inputs=[load_node])

            save_task = SaveTask(
                output_path,
                filesystem=self.storage.filesystem,
                features=features,
                overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            )
            save_node = EONode(save_task, inputs=[slice_node])
            processing_nodes.append(save_node)

        return EOWorkflow.from_endnodes(*processing_nodes)

    def get_execution_arguments(  # type: ignore[override]
        self, workflow: EOWorkflow, bbox_splits: List[Tuple[NamedBBox, List[NamedBBox]]]
    ) -> List[Dict[EONode, Dict[str, object]]]:
        nodes = workflow.get_nodes()
        load_node = nodes[0]
        save_nodes = [node for node in nodes if isinstance(node.task, SaveTask)]
        slice_nodes = [save_node.inputs[0] for save_node in save_nodes]

        exec_args: List[Dict[EONode, Dict[str, object]]] = []
        for (orig_name, _), split_bboxes in bbox_splits:
            single_exec: Dict[EONode, Dict[str, object]] = {load_node: dict(eopatch_folder=orig_name)}

            for i, (subbox_name, subbox) in enumerate(split_bboxes):
                single_exec[slice_nodes[i]] = dict(bbox=subbox)
                single_exec[save_nodes[i]] = dict(eopatch_folder=subbox_name)

            exec_args.append(single_exec)

        return exec_args

    def _get_features(self) -> List[FeatureSpec]:
        """Provides features that will be transformed by the pipeline."""
        meta_features = [FeatureType.BBOX]
        if any(f_type.is_temporal() for f_type, _ in self.config.features):
            meta_features += [FeatureType.TIMESTAMP]

        return self.config.features + meta_features

    def save_new_grid(self, bbox_splits: List[Tuple[NamedBBox, List[NamedBBox]]]) -> None:
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

                crs_grid = gpd.GeoDataFrame({"eopatch_names": names}, geometry=geometries, crs=crs.pyproj_crs())
                crs_grid.to_file(
                    local_file.path,
                    driver="GPKG",
                    encoding="utf-8",
                    layer=f"Grid EPSG:{crs_grid.crs.to_epsg()}",
                    engine=self.storage.config.geopandas_backend,
                )
