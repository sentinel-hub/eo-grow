import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "sampling")


@pytest.mark.order(after="test_rasterize.py::test_rasterize_pipeline")
@pytest.mark.parametrize(
    "experiment_name",
    [
        "sampling_fraction",
        "sampling_block_number",
        "sampling_block_fraction",
        "sampling_grid",
        pytest.param("sampling_chain", marks=pytest.mark.chain),
    ],
)
def test_sampling_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
