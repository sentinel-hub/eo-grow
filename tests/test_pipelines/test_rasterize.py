"""
Testing rasterization pipeline
"""
import os

import pytest
import geopandas as gpd
import shutil
from eolearn.core import FeatureType, EOPatch, MapFeatureTask, EONode

from eogrow.core.config import Config
from eogrow.utils.testing import ContentTester, check_pipeline_logs, run_and_test_pipeline
from eogrow.pipelines.rasterize import RasterizePipeline


class CropPreprocessTask(MapFeatureTask):
    """Applies mapping between names and crop ids"""

    CROP_MAP = {"wheat": 1, "barley": 4, "cotton": 5, "sugar beet": 13, "grape": 14}

    def map_method(self, dataframe, config):
        dataframe[config.values_column] = dataframe["CROP_TYPE"].apply(lambda crop_name: self.CROP_MAP[crop_name])

        return dataframe


class CropRasterizePipeline(RasterizePipeline):
    """A custom rasterization pipeline for crops"""

    def get_prerasterization_node(self, previous_node: EONode) -> EONode:
        """Applies mapping between names and crop ids"""
        return EONode(
            CropPreprocessTask(self.vector_feature, self.vector_feature, config=self.config), inputs=[previous_node]
        )


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

    for eopatch_name in pipeline.patch_list:
        eopatch_folder = os.path.join(pipeline.storage.get_folder("reference", full_path=True), eopatch_name)

        eopatch = EOPatch.load(eopatch_folder, lazy_loading=True)
        eopatch.vector_timeless["LULC_VECTOR"] = vector_data

        eopatch.save(eopatch_folder, features=[(FeatureType.VECTOR_TIMELESS, "LULC_VECTOR")], overwrite_permission=1)


@pytest.mark.order(before="test_rasterize_pipeline_features")
@pytest.mark.parametrize(
    "config_name, stats_name",
    [
        ("rasterize_pipeline_features_eroded.json", "rasterize_pipeline_features_eroded.json"),
        ("rasterize_pipeline_single_value.json", "rasterize_pipeline_single_value.json"),
    ],
)
def test_rasterize_pipeline(config_folder, config_name, stats_folder, stats_name):
    config = Config.from_path(os.path.join(config_folder, config_name))
    run_and_test_pipeline(config_folder, config_name, stats_folder, stats_name, config.output_folder_key)


@pytest.mark.chain
@pytest.mark.order(before="test_rasterize_pipeline_features")
def test_custom_rasterize_pipeline(config_folder):

    config_filename = os.path.join(config_folder, "rasterize_pipeline_config.json")

    pipeline = CropRasterizePipeline(Config.from_path(config_filename))

    # reset reference folder
    shutil.rmtree(pipeline.storage.get_folder("reference", full_path=True))

    pipeline.run()
    check_pipeline_logs(pipeline)


@pytest.mark.chain
def test_rasterize_pipeline_features(config_folder, stats_folder):
    config_filename = os.path.join(config_folder, "rasterize_pipeline_features.json")
    stat_path = os.path.join(stats_folder, "rasterize_stats.json")

    pipeline = RasterizePipeline(Config.from_path(config_filename))
    add_vector_data(pipeline)

    pipeline.run()
    check_pipeline_logs(pipeline)

    filesystem = pipeline.storage.filesystem
    folder = pipeline.storage.get_folder("reference")
    tester = ContentTester(filesystem, folder)
    # tester.save(stat_path)
    assert tester.compare(stat_path) == {}
