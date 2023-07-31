import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.order(after=["test_rasterize.py::test_rasterize_pipeline"])
@pytest.mark.parametrize("experiment_name", [pytest.param("zipmap", marks=pytest.mark.chain)])
def test_zipmap_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("zipmap", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path)
