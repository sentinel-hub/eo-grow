import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.order(after="test_download.py::test_download_pipeline")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "export_maps_mask",
        "export_maps_mask_local_copy",
        "export_maps_data",
        pytest.param("export_maps_data_compressed", marks=pytest.mark.chain),
    ],
)
def test_export_maps_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("export_maps", experiment_name)
    run_config(config_path)
    compare_content(config_path, stats_path)
