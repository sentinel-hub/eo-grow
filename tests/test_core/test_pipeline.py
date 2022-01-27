"""
Testing basic pipeline functionalities
"""
import os
import logging

import pytest

from eolearn.core import EOWorkflow

from eogrow.core.pipeline import Pipeline
from eogrow.core.config import interpret_config_from_path

pytestmark = pytest.mark.fast


@pytest.fixture(scope="session", name="simple_config_filename")
def simple_config_filename_fixture(config_folder):
    return os.path.join(config_folder, "simple_config.json")


class SimplePipeline(Pipeline):
    class Schema(Pipeline.Schema):
        test_param: int

    def run_procedure(self):
        logger = logging.getLogger(__name__)
        logger.debug("Some log")

        workflow = EOWorkflow([])
        exec_args = self.get_execution_arguments(workflow)

        finished, failed, _ = self.run_execution(workflow, exec_args)

        return finished[:-1], finished[-1:] + failed


def test_pipeline_execution(simple_config_filename):

    pipeline = SimplePipeline(interpret_config_from_path(simple_config_filename))
    pipeline.run()

    logs_folder = pipeline.logging_manager.get_pipeline_logs_folder(pipeline.current_execution_name, full_path=True)
    assert os.path.isdir(logs_folder)

    for filename in ["failed.json", "finished.json", "pipeline-report.json", "pipeline.log"]:
        assert os.path.isfile(os.path.join(logs_folder, filename))

    folder_content = sorted(os.listdir(logs_folder))
    assert len(folder_content) == 5
    assert folder_content[0].startswith("eoexecution-report")
