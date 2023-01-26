import os

import geopandas as gpd
import pytest

from eolearn.core import EONode, EOPatch, FeatureType, MapFeatureTask

from eogrow.core.config import interpret_config_from_path
from eogrow.pipelines.rasterize import RasterizePipeline
from eogrow.utils.testing import ContentTester, check_pipeline_logs, create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "rasterize")


class CropPreprocessTask(MapFeatureTask):
    """Applies mapping between names and crop ids"""

    CROP_MAP = {"wheat": 1, "barley": 4, "cotton": 5, "sugar beet": 13, "grape": 14}

    def map_method(self, dataframe, config):
        crop_column = config.columns[0].values_column

        dataframe[crop_column] = dataframe["CROP_TYPE"].apply(lambda crop_name: self.CROP_MAP[crop_name])

        return dataframe


class CropRasterizePipeline(RasterizePipeline):
    """A custom rasterization pipeline for crops"""

    def preprocess_dataset(self, dataframe):
        """Adds polygon ids"""
        dataframe["POLYGON_ID"] = dataframe.index + 1
        return dataframe

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

    for eopatch_name, _ in pipeline.get_patch_list():
        eopatch_folder = os.path.join(pipeline.storage.get_folder("reference", full_path=True), eopatch_name)

        eopatch = EOPatch.load(eopatch_folder, lazy_loading=True)
        eopatch.vector_timeless["LULC_VECTOR"] = vector_data

        eopatch.save(eopatch_folder, features=[(FeatureType.VECTOR_TIMELESS, "LULC_VECTOR")], overwrite_permission=1)


@pytest.mark.parametrize("experiment_name", ["rasterize_pipeline_float"])
def test_rasterize_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)


@pytest.mark.chain
@pytest.mark.order(before="test_rasterize_pipeline_features")
@pytest.mark.parametrize("experiment_name", ["rasterize_pipeline"])
def test_rasterize_pipeline_preprocess(folders, experiment_name):
    # Can't use utility testing due to custom pipeline
    config_filename = os.path.join(folders["config_folder"], experiment_name + ".json")
    stat_path = os.path.join(folders["stats_folder"], experiment_name + ".json")

    config = interpret_config_from_path(config_filename)
    pipeline = CropRasterizePipeline.from_raw_config(config)

    filesystem = pipeline.storage.filesystem
    folder = pipeline.storage.get_folder(pipeline.config.output_folder_key)
    filesystem.removetree(folder)

    pipeline.run()
    check_pipeline_logs(pipeline)

    tester = ContentTester(filesystem, folder)
    # tester.save(stat_path)
    assert tester.compare(stat_path) == {}


@pytest.mark.chain
@pytest.mark.parametrize("experiment_name", ["rasterize_pipeline_features"])
def test_rasterize_pipeline_features(folders, experiment_name):
    config_filename = os.path.join(folders["config_folder"], experiment_name + ".json")
    stat_path = os.path.join(folders["stats_folder"], experiment_name + ".json")

    config = interpret_config_from_path(config_filename)
    pipeline = RasterizePipeline.from_raw_config(config)
    add_vector_data(pipeline)

    pipeline.run()
    check_pipeline_logs(pipeline)

    filesystem = pipeline.storage.filesystem
    folder = pipeline.storage.get_folder(pipeline.config.output_folder_key)
    tester = ContentTester(filesystem, folder)
    # tester.save(stat_path)
    assert tester.compare(stat_path) == {}
