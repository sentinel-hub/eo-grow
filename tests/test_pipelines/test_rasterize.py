import os
from typing import Optional

import geopandas as gpd
import pytest

from eolearn.core import EONode, EOPatch, FeatureType, MapFeatureTask

from eogrow.core.config import collect_configs_from_path, interpret_config_from_dict
from eogrow.pipelines.rasterize import RasterizePipeline
from eogrow.utils.meta import load_pipeline_class
from eogrow.utils.testing import check_pipeline_logs, compare_content, run_config

pytestmark = pytest.mark.integration


class CropPreprocessTask(MapFeatureTask):
    """Applies mapping between names and crop ids"""

    CROP_MAP = {"wheat": 1, "barley": 4, "cotton": 5, "sugar beet": 13, "grape": 14}

    def map_method(self, dataframe):
        dataframe["CROP_ID"] = dataframe["CROP_TYPE"].apply(lambda crop_name: self.CROP_MAP[crop_name])
        return dataframe


class CropRasterizePipeline(RasterizePipeline):
    """A custom rasterization pipeline for crops"""

    def preprocess_dataset(self, dataframe):
        """Adds polygon ids"""
        dataframe["POLYGON_ID"] = dataframe.index + 1
        return dataframe

    def get_prerasterization_node(self, previous_node: EONode) -> EONode:
        """Applies mapping between names and crop ids"""
        return EONode(CropPreprocessTask(self.vector_feature, self.vector_feature), inputs=[previous_node])


def add_vector_data(pipeline):
    """Adds a vector feature to existing eopatches"""
    LULC_MAP = {
        "no data": 0,
        "cultivated land": 1,
        "grassland": 3,
        "shrubland": 4,
        "water": 5,
        "artificial surface": 8,
        "bareland": 9,
    }
    vector_data = os.path.join(pipeline.storage.get_input_data_folder(full_path=True), "test_area_lulc.geojson")
    vector_data = gpd.read_file(vector_data, encoding="utf-8")

    vector_data["LULC_ID"] = vector_data["LULC"].apply(lambda lulc_name: LULC_MAP[lulc_name])
    vector_data["LULC_POLYGON_ID"] = vector_data.index + 1

    for eopatch_name, _ in pipeline.get_patch_list():
        eopatch_folder = os.path.join(pipeline.storage.get_folder("reference", full_path=True), eopatch_name)

        eopatch = EOPatch.load(eopatch_folder, lazy_loading=True)
        eopatch.vector_timeless["LULC_VECTOR"] = vector_data

        eopatch.save(eopatch_folder, features=[(FeatureType.VECTOR_TIMELESS, "LULC_VECTOR")], overwrite_permission=1)


def reset_output_folder(config_path: str) -> Optional[str]:
    raw_config = interpret_config_from_dict(collect_configs_from_path(config_path)[0])
    pipeline = load_pipeline_class(raw_config).from_raw_config(raw_config)

    output_folder_key: Optional[str] = raw_config.get("output_folder_key")
    if output_folder_key is not None:
        folder = pipeline.storage.get_folder(output_folder_key)
        pipeline.storage.filesystem.removetree(folder)
        return pipeline.storage.get_folder(output_folder_key, full_path=True)


@pytest.mark.parametrize("experiment_name", ["rasterize_pipeline_float"])
def test_rasterize_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("rasterize", experiment_name)
    reset_output_folder(config_path)

    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)


@pytest.mark.chain
@pytest.mark.order(before="test_rasterize_pipeline_features")
@pytest.mark.parametrize("experiment_name", ["rasterize_pipeline"])
def test_rasterize_pipeline_preprocess(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("rasterize", experiment_name)
    crude_configs = collect_configs_from_path(config_path)
    raw_configs = [interpret_config_from_dict(config) for config in crude_configs]

    output_folder = reset_output_folder(config_path)
    for config in raw_configs:
        pipeline = CropRasterizePipeline.from_raw_config(config)
        pipeline.run()
        check_pipeline_logs(pipeline)

    compare_content(output_folder, stats_path)


@pytest.mark.chain
@pytest.mark.parametrize("experiment_name", ["rasterize_pipeline_features"])
def test_rasterize_pipeline_features(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("rasterize", experiment_name)
    raw_config = interpret_config_from_dict(collect_configs_from_path(config_path)[0])

    first_pipeline = load_pipeline_class(raw_config).from_raw_config(raw_config)
    add_vector_data(first_pipeline)

    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)
