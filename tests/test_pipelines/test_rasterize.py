import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("experiment_name", ["rasterize_file"])
def test_rasterize_file(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("rasterize", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path)


@pytest.mark.chain()
@pytest.mark.order(before="test_rasterize_feature_with_resolution")
@pytest.mark.parametrize(("preparation_config", "config"), [("load_crops_vector_data", "rasterize_feature_with_shape")])
def test_rasterize_feature_with_shape(config_and_stats_paths, preparation_config, config):
    preparation_config_path, _ = config_and_stats_paths("rasterize", preparation_config)
    config_path, stats_path = config_and_stats_paths("rasterize", config)

    run_config(preparation_config_path)
    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)


@pytest.mark.chain()
@pytest.mark.parametrize(
    ("preparation_config", "config"), [("load_lulc_vector_data", "rasterize_feature_with_resolution")]
)
def test_rasterize_feature_with_resolution(config_and_stats_paths, preparation_config, config):
    preparation_config_path, _ = config_and_stats_paths("rasterize", preparation_config)
    config_path, stats_path = config_and_stats_paths("rasterize", config)

    run_config(preparation_config_path, reset_output_folder=False)
    output_path = run_config(config_path, reset_output_folder=False)
    compare_content(output_path, stats_path)
