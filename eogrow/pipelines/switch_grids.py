"""
Pipelines for testing
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, root_validator

from eolearn.core import EONode, EOWorkflow, FeatureType, LoadTask, OverwritePermission, SaveTask
from sentinelhub import BBox

from ..core.area.base import AreaManager
from ..core.eopatch import EOPatchManager
from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema, ManagerSchema
from ..tasks.spatial import SpatialJoinTask, SpatialSliceTask
from ..utils.grid import GridTransformation
from ..utils.types import Feature, FeatureSpec
from ..utils.validators import field_validator, optional_field_validator, validate_manager

LOGGER = logging.getLogger(__name__)


class FeatureSchema(BaseSchema):
    feature: FeatureSpec = Field(description="A feature to be processed.")
    no_data_value: float = Field(
        0,
        description=(
            "Used for spatial raster features to fill out parts of a target grid that are outside of a source grid."
        ),
    )
    unique_columns: Optional[List[str]] = Field(
        description=(
            "This is only meant to be used for vector features. If provided, it will use this list of columns to drop"
            " duplicated rows in spatially joined dataframes."
        )
    )

    @root_validator
    def check_values(cls, values):  # type: ignore
        """Multiple different checks that given values make sense."""
        feature = values["feature"]
        feature_type = feature if isinstance(feature, FeatureType) else feature[0]

        assert feature_type is not FeatureType.BBOX, "A bounding box doesn't have to be specified in config"

        assert (
            feature is FeatureType.TIMESTAMP or feature != feature_type
        ), "All features, except timestamp, should have feature name"

        assert (
            values.get("no_data_value") == 0 or feature_type.is_raster()
        ), "Parameter 'no_data_value' can only be set for raster feature types"

        assert (
            values.get("unique_columns") is None or feature_type.is_vector()
        ), "Parameter 'unique_column' can only be set for vector feature types"

        return values


class SwitchGridsPipeline(Pipeline):
    """Pipeline that converts one EOPatch tiling grid into another.

    Notes:
      - Grid transformations are defined in area managers. A pair of grids are compatible if and only if the source
        area manager implements a transformation into the target area manager.
      - By default, if a grid transformation would cause a misalignment in any EOPatch feature, an error would be
        raised during EOWorkflow execution. Make sure to choose parameters where that won't happen.
    """

    class Schema(Pipeline.Schema):
        target_area: ManagerSchema = Field(description="An area manager configuration that defines a new grid.")
        validate_area = field_validator("target_area", validate_manager, pre=True)

        target_eopatch: Optional[ManagerSchema] = Field(
            description=(
                "An EOPatch manager configuration that defines naming in the new grid. If not provided, the old naming"
                " will be used."
            )
        )
        validate_eopatch = optional_field_validator("target_eopatch", validate_manager, pre=True)

        input_folder_key: str = Field(
            description="A storage manager key pointing to the folder from where data will be loaded."
        )
        output_folder_key: str = Field(
            description="A storage manager key pointing to the folder to where data will be saved."
        )
        features: List[FeatureSchema] = Field(description="Which features will be loaded and joined.")
        raise_misaligned: bool = Field(
            True,
            description=(
                "If True this will raise an error if splitting or joining any spatial raster EOPatch feature would "
                "cause a misalignment. If False, misalignment issues will be ignored."
            ),
        )

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.target_area_manager: AreaManager = self._load_manager(self.config.target_area, storage=self.storage)

        target_eopatch_config = self.config.target_eopatch or self.config.eopatch
        self.target_eopatch_manager: EOPatchManager = self._load_manager(
            target_eopatch_config, area_manager=self.target_area_manager
        )

    def run_procedure(self) -> Tuple[List, List]:
        """A procedure that defines a transformation between grids and transforms EOPatches."""
        transformations = self.area_manager.transform_grid(self.target_area_manager)
        if not transformations:
            return [], []

        workflow = self.build_workflow(transformations)
        exec_args = self.get_execution_arguments_from_transformations(workflow, transformations)
        execution_names, name_mapping = self._get_execution_names_and_mapping(transformations)

        finished, failed, _ = self.run_execution(workflow, exec_args, eopatch_list=execution_names)

        finished = [input_name for execution_name in finished for input_name in name_mapping[execution_name]]
        failed = [input_name for execution_name in failed for input_name in name_mapping[execution_name]]
        return finished, failed

    def build_workflow(self, transformations: List[GridTransformation]) -> EOWorkflow:
        """Create a workflow that consists of loading, joining groups of EOPatches, slicing into new EOPatches and
        saving them."""
        features = self._get_features()

        input_path = self.storage.get_folder(self.config.input_folder_key, full_path=True)
        max_input_patch_num = max(len(transformation.source_bboxes) for transformation in transformations)
        load_nodes = [EONode(LoadTask(input_path, config=self.sh_config)) for _ in range(max_input_patch_num)]

        join_task = SpatialJoinTask(
            features,
            no_data_map=self._get_no_data_map(),
            unique_columns_map=self._get_unique_columns_map(),
            raise_misaligned=self.config.raise_misaligned,
        )
        join_node = EONode(join_task, inputs=load_nodes)

        save_nodes = []
        output_path = self.storage.get_folder(self.config.output_folder_key, full_path=True)
        max_output_patch_num = max(len(transformation.target_bboxes) for transformation in transformations)
        for _ in range(max_output_patch_num):
            slice_task = SpatialSliceTask(features, raise_misaligned=self.config.raise_misaligned)
            slice_node = EONode(slice_task, inputs=[join_node])

            save_task = SaveTask(
                output_path,
                features=features,
                overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                config=self.sh_config,
            )
            save_node = EONode(save_task, inputs=[slice_node])
            save_nodes.append(save_node)

        return EOWorkflow.from_endnodes(*save_nodes)

    def get_execution_arguments_from_transformations(
        self, workflow: EOWorkflow, transformations: List[GridTransformation]
    ) -> List[Dict[EONode, Dict[str, object]]]:
        """Creates execution arguments."""
        nodes = workflow.get_nodes()
        load_nodes = [node for node in nodes if isinstance(node.task, LoadTask)]
        save_nodes = [node for node in nodes if isinstance(node.task, SaveTask)]
        join_node = [node for node in nodes if isinstance(node.task, SpatialJoinTask)][0]
        slice_nodes = [save_node.inputs[0] for save_node in save_nodes]

        return [
            {
                join_node: dict(bbox=transformation.enclosing_bbox),
                **self._get_io_exec_dict(transformation, load_nodes=load_nodes, save_nodes=save_nodes),
                **self._get_slice_exec_dict(transformation, slice_nodes),
            }
            for transformation in transformations
        ]

    def _get_io_exec_dict(
        self, transformation: GridTransformation, *, load_nodes: List[EONode], save_nodes: List[EONode]
    ) -> Dict[EONode, Dict[str, Any]]:
        """Prepares execution arguments for load and save nodes."""
        input_names = list(self.eopatch_manager.generate_names(transformation.source_df))
        input_names += [None] * (len(load_nodes) - len(input_names))

        output_names = list(self.target_eopatch_manager.generate_names(transformation.target_df))
        output_names += [None] * (len(save_nodes) - len(output_names))

        return {
            node: dict(eopatch_folder=name) for name, node in zip(input_names + output_names, load_nodes + save_nodes)
        }

    @staticmethod
    def _get_slice_exec_dict(
        transformation: GridTransformation, slice_nodes: List[EONode]
    ) -> Dict[EONode, Dict[str, Any]]:
        """Prepares execution arguments for slice task nodes."""
        slice_bboxes: List[Optional[BBox]] = list(transformation.target_bboxes)
        slice_bboxes += [None] * (len(slice_nodes) - len(transformation.target_bboxes))

        return {node: dict(bbox=bbox) for bbox, node in zip(slice_bboxes, slice_nodes)}

    def _get_execution_names_and_mapping(
        self, transformations: List[GridTransformation]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """An execution name will always be the name of alphabetically first name of input EOPatches. That is because
        each output EOPatch should be in exactly 1 transformation while an input EOPatch can be used in multiple
        transformations. A mapping will be provided from an execution name to a list of input EOPatches."""
        names: List[str] = []
        mapping: Dict[str, List[str]] = {}

        for transformation in transformations:
            input_names = list(self.eopatch_manager.generate_names(transformation.source_df))
            output_names = list(self.target_eopatch_manager.generate_names(transformation.target_df))
            execution_name = min(output_names)

            names.append(execution_name)
            if execution_name in mapping:
                raise ValueError(f"An output EOPatch {execution_name} is being used in more than 1 transformation.")
            mapping[execution_name] = input_names

        return names, mapping

    def _get_features(self) -> List[FeatureSpec]:
        """Provides features that will be transformed by the pipeline."""
        features = [feature_config.feature for feature_config in self.config.features]
        return features + [FeatureType.BBOX]

    def _get_no_data_map(self) -> Dict[Feature, float]:
        """Provides a map between spatial raster features and their 'no data' values."""
        no_data_map: Dict[Feature, float] = {}
        for feature_config in self.config.features:
            feature = feature_config.feature
            feature_type = feature if isinstance(feature, FeatureType) else feature[0]

            if feature_type.is_spatial() and feature_type.is_raster():
                no_data_map[feature] = feature_config.no_data_value

        return no_data_map

    def _get_unique_columns_map(self) -> Dict[Feature, List[str]]:
        """Provides a map between vector raster features and their unique columns."""
        unique_columns_map: Dict[Feature, List[str]] = {}
        for feature_config in self.config.features:
            if feature_config.unique_columns:
                unique_columns_map[feature_config.feature] = feature_config.unique_columns
        return unique_columns_map
