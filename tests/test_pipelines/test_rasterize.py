import pytest

from eogrow.core.config import collect_configs_from_path, interpret_config_from_dict
from eogrow.utils.meta import load_pipeline_class
from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


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
@pytest.mark.parametrize("preparation_config, config", [("prepare_vector_data", "rasterize_pipeline_vector_feature")])
def test_rasterize_pipeline_vector_feature(config_and_stats_paths, preparation_config, config):
    preparation_config_path, _ = config_and_stats_paths("rasterize", preparation_config)
    config_path, stats_path = config_and_stats_paths("rasterize", config)

    run_config(preparation_config_path)
    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)
