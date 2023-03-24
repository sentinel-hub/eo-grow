import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "import_vector")


@pytest.mark.parametrize("experiment_name", ["import_vector", "import_vector_temporal"])
def test_import_tiff_pipeline(folders, experiment_name):
    run_and_test_pipeline(experiment_name, **folders)
