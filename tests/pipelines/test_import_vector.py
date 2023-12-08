import pytest

from eogrow.utils.testing import compare_content, run_config

pytestmark = pytest.mark.integration


@pytest.mark.chain()
@pytest.mark.parametrize("experiment_name", ["import_vector", "import_vector_temporal"])
def test_import_vector_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("import_vector", experiment_name)
    output_path = run_config(config_path)
    compare_content(output_path, stats_path, save_new_stats=True)
