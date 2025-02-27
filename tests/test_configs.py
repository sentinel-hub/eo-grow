import os
import re
from glob import glob

import pytest

import eogrow
from eogrow.core.config import collect_configs_from_path, interpret_config_from_dict
from eogrow.utils.meta import load_pipeline_class

IGNORED_FOLDERS = ["other"]
CONFIG_REGEX = re.compile(r"^(?!global_).*\.json$")


def pytest_generate_tests(metafunc):
    files = glob(os.path.join(eogrow.__path__[0], "..", "tests", "test_config_files", "**", "*.json"), recursive=True)
    files = [conf_path for conf_path in files if CONFIG_REGEX.match(os.path.split(conf_path)[-1])]
    files = [conf_path for conf_path in files if not any(folder in conf_path for folder in IGNORED_FOLDERS)]
    metafunc.parametrize("config_file", files, ids=[path.split("test_config_files/")[-1] for path in files])


def test_validate_configs(config_file):
    crude_config = collect_configs_from_path(config_file)
    if isinstance(crude_config, list):
        crude_configs = [conf["pipeline_config"] for conf in crude_config]
    else:
        crude_configs = [crude_config]

    for crude_config in crude_configs:
        pipeline = crude_config.get("pipeline")
        if pipeline is None:
            pytest.skip(f"Config with pipeline {pipeline} will be ignored. Skipping.")

        config = interpret_config_from_dict(crude_config)
        load_pipeline_class(config).Schema.parse_obj(config)
