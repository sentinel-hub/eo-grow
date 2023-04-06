import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.order(after=["test_prediction.py::test_rasterization_pipeline"])
@pytest.mark.parametrize("experiment_name", [pytest.param("zipmap", marks=pytest.mark.chain)])
def test_zipmap_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("zipmap", experiment_name)
    run_config(config_path)
    compare_content(config_path, stats_path)
