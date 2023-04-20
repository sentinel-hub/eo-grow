import pytest

from eolearn.core import EONode, MapFeatureTask

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


def reset_output_folder(config_path: str) -> str:
    raw_config = interpret_config_from_dict(collect_configs_from_path(config_path)[0])
    pipeline = load_pipeline_class(raw_config).from_raw_config(raw_config)

    output_folder_key: str = raw_config["output_folder_key"]
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
@pytest.mark.order(before="test_rasterize_pipeline_vector_feature")
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
@pytest.mark.parametrize("preparation_config, config", [("prepare_vector_data", "rasterize_pipeline_vector_feature")])
def test_rasterize_pipeline_vector_feature(config_and_stats_paths, preparation_config, config):
    preparation_config_path, _ = config_and_stats_paths("rasterize", preparation_config)
    config_path, stats_path = config_and_stats_paths("rasterize", config)

    run_config(preparation_config_path, reset_output_folder=False)
    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)
