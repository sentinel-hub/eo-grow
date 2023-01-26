import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "zipmap")


@pytest.mark.order(after=["test_prediction.py::test_rasterization_pipeline"])
@pytest.mark.parametrize("experiment_name", [pytest.param("zipmap", marks=pytest.mark.chain)])
def test_zipmap_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
