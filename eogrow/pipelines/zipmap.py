import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set

from pydantic import Field, validator

from eolearn.core import (
    EONode,
    EOWorkflow,
    FeatureType,
    LoadTask,
    MergeEOPatchesTask,
    OverwritePermission,
    SaveTask,
    ZipFeatureTask,
)

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..types import Feature, FeatureSpec, PatchList
from ..utils.filter import get_patches_with_missing_features
from ..utils.meta import import_object

LOGGER = logging.getLogger(__name__)


class InputFeatureSchema(BaseSchema):
    feature: Feature = Field(description="Which features to load from folder.")
    folder_key: str = Field(description="The storage manager key pointing to the folder from which to load data.")
    include_bbox_and_timestamp = Field(
        True,
        description="Auto loads BBOX and (if the features is temporal) TIMESTAMP.",
    )


class ZipMapPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        input_features: List[InputFeatureSchema] = Field(
            description="The specification for all the features to be loaded."
        )

        zipmap_import_path: str = Field(
            description="Import path of the callable with which to process the loaded features."
        )
        params_model: Optional[str] = Field(
            description=(
                "Optional import path for the pydantic model class, with which to parse and validate the parameters for"
                " the callable. The model will be used to parse the params and then unpacked back into a dictionary, "
                " which is passed to the callable as `**params`."
            )
        )
        params: Dict[str, Any] = Field(
            default_factory=dict, description="Any keyword arguments to be passed to the zipmap function."
        )

        @validator("params")
        def parse_params(cls, v: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
            """Parse the parameters according to model, but returning as a dictionary to allow `**kwargs` passing."""
            if values.get("params_model"):
                params_model: BaseSchema = import_object(values["params_model"])
                return params_model.parse_obj(v).dict()
            return v

        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the algorithm pipeline."
        )
        output_feature: Feature

        compress_level: int = Field(1, description="Level of compression used in saving eopatches.")

    config: Schema

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of new features"""
        # Note: does not catch missing BBox or Timestamp
        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            [self.config.output_feature],
        )

        return filtered_patch_list

    def get_load_nodes(self) -> List[EONode]:
        """Prepare all nodes with load tasks."""
        load_schema: DefaultDict[str, Set[FeatureSpec]] = defaultdict(set)
        for input_feature in self.config.input_features:
            features_to_load = load_schema[input_feature.folder_key]
            features_to_load.add(input_feature.feature)

            if input_feature.include_bbox_and_timestamp:
                features_to_load.add(FeatureType.BBOX)
                if input_feature.feature[0].is_temporal():
                    features_to_load.add(FeatureType.TIMESTAMPS)

        load_nodes = []
        for folder_key, features in load_schema.items():
            folder_path = self.storage.get_folder(folder_key, full_path=True)
            load_nodes.append(
                EONode(LoadTask(folder_path, config=self.sh_config, features=list(features), lazy_loading=True))
            )

        return load_nodes

    def get_zipmap_node(self, previous_node: EONode) -> EONode:
        """Adds all algorithm and dataframe-saving nodes and returns the endnode."""

        zipmap = import_object(self.config.zipmap_import_path)
        input_features = [input_schema.feature for input_schema in self.config.input_features]
        zipmap_task = ZipFeatureTask(input_features, self.config.output_feature, zipmap, **self.config.params)

        return EONode(zipmap_task, inputs=[previous_node])

    def build_workflow(self) -> EOWorkflow:
        """Builds the workflow"""

        load_nodes = self.get_load_nodes()
        merge_node = EONode(MergeEOPatchesTask(), inputs=load_nodes)

        mapping_node = self.get_zipmap_node(merge_node)

        save_path = self.storage.get_folder(self.config.output_folder_key, full_path=True)
        save_task = SaveTask(
            save_path,
            config=self.sh_config,
            features=[self.config.output_feature, FeatureType.BBOX, FeatureType.TIMESTAMPS],
            compress_level=self.config.compress_level,
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
        )
        save_node = EONode(save_task, inputs=[mapping_node])

        return EOWorkflow.from_endnodes(save_node)
