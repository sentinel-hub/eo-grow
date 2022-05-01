"""
Tests for testing pipelines
"""

import pytest

from eogrow.utils.testing import create_folder_dict, run_and_test_pipeline

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "testing")


@pytest.mark.parametrize(
    "experiment_name",
    ["testing"],
)
def test_features_pipeline(experiment_name, folders):
    run_and_test_pipeline(experiment_name, **folders)
