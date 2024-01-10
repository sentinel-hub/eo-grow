import logging
import os
from typing import List, Tuple, Union

import pytest

from eolearn.core import EOWorkflow
from sentinelhub import CRS, BBox

from eogrow.core.config import interpret_config_from_path
from eogrow.core.pipeline import Pipeline, PipelineExecutionError


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


class FailingPipeline(Pipeline):
    class Schema(Pipeline.Schema):
        fail: bool

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        return [], ([0] if self.config.fail else [])


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


@pytest.mark.parametrize(
    ("test_subset", "expected_result"),
    [
        ([], []),
        (
            [0, "eopatch-id-1-col-0-row-1"],
            [
                ("eopatch-id-0-col-0-row-0", BBox(((729480.0, 4390045.0), (732120.0, 4391255.0)), crs=CRS(32638))),
                ("eopatch-id-1-col-0-row-1", BBox(((729480.0, 4391145.0), (732120.0, 4392355.0)), crs=CRS(32638))),
            ],
        ),
        (
            [0, 1],
            [
                ("eopatch-id-0-col-0-row-0", BBox(((729480.0, 4390045.0), (732120.0, 4391255.0)), crs=CRS(32638))),
                ("eopatch-id-1-col-0-row-1", BBox(((729480.0, 4391145.0), (732120.0, 4392355.0)), crs=CRS(32638))),
            ],
        ),
    ],
)
def test_get_patch_list_filtration(
    test_subset: List[Union[int, str]], expected_result: List[Tuple[str, BBox]], simple_config_filename: str
) -> None:
    """Tests that the `test_subset` filtration is done correctly."""
    config = interpret_config_from_path(simple_config_filename)
    config["test_subset"] = test_subset
    pipeline = SimplePipeline.from_raw_config(config)
    assert pipeline.get_patch_list() == expected_result


@pytest.mark.parametrize("test_subset", [[0, 100], ["beep"]])
def test_get_patch_list_filtration_error(test_subset: List[Union[int, str]], simple_config_filename: str) -> None:
    """Tests that the `test_subset` filtration fails if indices or names are not valid."""
    config = interpret_config_from_path(simple_config_filename)
    config["test_subset"] = test_subset
    pipeline = SimplePipeline.from_raw_config(config)
    with pytest.raises(ValueError):
        pipeline.get_patch_list()


@pytest.mark.parametrize("fail", [True, False])
@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_pipeline_raises_on_failure(fail: bool, raise_on_failure: bool, simple_config_filename: str):
    config = interpret_config_from_path(simple_config_filename)
    config.pop("test_param")
    pipeline = FailingPipeline.from_raw_config({**config, "fail": fail, "raise_on_failure": raise_on_failure})

    if fail and raise_on_failure:
        with pytest.raises(PipelineExecutionError):
            pipeline.run()
    else:
        pipeline.run()
