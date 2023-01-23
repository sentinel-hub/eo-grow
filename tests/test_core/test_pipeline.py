import logging
import os
from typing import List, Tuple

import pytest

from eolearn.core import EOWorkflow
from sentinelhub import CRS, BBox

from eogrow.core.config import interpret_config_from_path
from eogrow.core.pipeline import Pipeline

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="simple_config_filename")
def simple_config_filename_fixture(config_folder):
    return os.path.join(config_folder, "other", "simple_config.json")


class SimplePipeline(Pipeline):
    class Schema(Pipeline.Schema):
        test_param: int

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        logger = logging.getLogger(__name__)
        logger.debug("Some log")

        workflow = EOWorkflow([])
        patch_list = self.get_patch_list()
        exec_args = self.get_execution_arguments(workflow, patch_list)

        finished, failed, _ = self.run_execution(workflow, exec_args)

        return finished[:-1], finished[-1:] + failed


def test_pipeline_execution(simple_config_filename: str) -> None:
    """Tests that appropriate folders and log files are created."""
    config = interpret_config_from_path(simple_config_filename)
    pipeline = SimplePipeline.from_raw_config(config)
    pipeline.run()

    logs_folder = pipeline.logging_manager.get_pipeline_logs_folder(pipeline.current_execution_name, full_path=True)
    assert os.path.isdir(logs_folder)

    for filename in ["failed.json", "finished.json", "pipeline-report.json", "pipeline.log"]:
        assert os.path.isfile(os.path.join(logs_folder, filename))

    folder_content = sorted(os.listdir(logs_folder))
    assert len(folder_content) == 5
    assert folder_content[0].startswith("eoexecution-report")


def test_get_patch_list_filtration(simple_config_filename: str) -> None:
    """Tests that the `test_subset` filtration is done correctly."""
    config = interpret_config_from_path(simple_config_filename)
    pipeline = SimplePipeline.from_raw_config(config)
    expected_patch_list = [
        ("eopatch-id-0-col-0-row-0", BBox(((729480.0, 4390045.0), (732120.0, 4391255.0)), crs=CRS(32638))),
        ("eopatch-id-1-col-0-row-1", BBox(((729480.0, 4391145.0), (732120.0, 4392355.0)), crs=CRS(32638))),
    ]
    assert pipeline.get_patch_list() == expected_patch_list
