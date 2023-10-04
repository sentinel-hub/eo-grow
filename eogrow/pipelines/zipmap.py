from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from eolearn.core import (
    EONode,
    EOWorkflow,
    LoadTask,
    MergeEOPatchesTask,
    OverwritePermission,
    SaveTask,
    ZipFeatureTask,
)
from eolearn.core.types import Feature

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..types import PatchList
from ..utils.filter import get_patches_with_missing_features
from ..utils.meta import import_object

LOGGER = logging.getLogger(__name__)


class InputFeatureSchema(BaseSchema):
    feature: Feature = Field(description="Which features to load from folder.")
    folder_key: str = Field(description="The storage manager key pointing to the folder from which to load data.")


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
        def parse_params(cls, v: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
            """Parse the parameters according to model, but returning as a dictionary to allow `**kwargs` passing."""
            if values.get("params_model"):
                params_model: BaseSchema = import_object(values["params_model"])
                return params_model.parse_obj(v).dict()
            return v

        output_folder_key: str = Field(
            description="The storage manager key pointing to the output folder for the algorithm pipeline."
        )
        output_feature: Feature

    config: Schema

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of new features"""
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            [self.config.output_feature],
            check_timestamps=self.config.output_feature[0].is_temporal(),
        )

    def get_load_nodes(self) -> list[EONode]:
        """Prepare all nodes with load tasks."""
        load_schema: defaultdict[str, set[Feature]] = defaultdict(set)
        for input_feature in self.config.input_features:
            load_schema[input_feature.folder_key].add(input_feature.feature)

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
            features=[self.config.output_feature],
            overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
            use_zarr=self.storage.config.use_zarr,
        )
        save_node = EONode(save_task, inputs=[mapping_node])

        return EOWorkflow.from_endnodes(save_node)
