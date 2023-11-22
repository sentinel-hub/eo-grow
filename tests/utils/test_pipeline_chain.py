import os

import pytest
from pydantic import ValidationError

from eogrow.core.config import collect_configs_from_path
from eogrow.utils.pipeline_chain import run_pipeline_chain, validate_pipeline_chain


@pytest.fixture(name="global_config")
def global_config_fixture(config_folder):
    return collect_configs_from_path(os.path.join(config_folder, "global_config.json"))


@pytest.fixture(name="some_valid_pipeline_config")
def some_valid_pipeline_config_fixture(global_config):
    return {
        "pipeline": "eogrow.pipelines.import_tiff.ImportTiffPipeline",
        "tiff_folder_key": "input_data",
        "output_folder_key": "temp",
        "output_feature": ["data", "ImportedData"],
        "input_filename": "import_test.tiff",
        **global_config,
    }


def test_validate_pipeline_chain(some_valid_pipeline_config):
    good_chain = [
        {"pipeline_config": some_valid_pipeline_config},
        {"pipeline_config": some_valid_pipeline_config, "pipeline_resources": {"num_cpus": 1}},
    ]

    validate_pipeline_chain(good_chain)


def test_validate_pipeline_chain_fail(some_valid_pipeline_config):
    bad_config_in_chain = [
        {"pipeline_config": {"nonexisting_param": "quack quack", **some_valid_pipeline_config}},
    ]
    with pytest.raises(ValidationError):
        validate_pipeline_chain(bad_config_in_chain)

    bad_chain = [
        {"pipeline_config": some_valid_pipeline_config},
        {"pipeline_config": some_valid_pipeline_config, "PipelineResources": {"num_cpus": 1}},
    ]
    with pytest.raises(TypeError):
        validate_pipeline_chain(bad_chain)


def test_run_pipeline_chain(global_config):
    pipeline_config = {
        "pipeline": "eogrow.pipelines.testing.GenerateDataPipeline",
        "output_folder_key": "temp",
        "seed": 42,
        "timestamps": {"time_period": ["2021-06-15", "2022-09-05"], "num_timestamps": 10},
        **global_config,
    }
    chain = [
        {"pipeline_config": pipeline_config},
        {"pipeline_config": pipeline_config, "pipeline_resources": {"num_cpus": 1}},
    ]
    run_pipeline_chain(chain)
