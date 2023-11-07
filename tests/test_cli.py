"""
Module for testing command line interface
"""

import subprocess

import pytest


@pytest.mark.parametrize(
    "command",
    [
        "eogrow",
        "eogrow-ray",
        "eogrow-template",
        "eogrow-validate",
        "eogrow-test",
    ],
)
def test_help(command):
    """Tests a simple execution from command line"""
    assert subprocess.call(f"{command} --help", shell=True) == 0


@pytest.mark.parametrize(
    "command_params",
    [
        "eogrow.core.storage.StorageManager --template-format open-api",
        "eogrow.core.area.batch.BatchAreaManager",
        "eogrow.pipelines.download.DownloadPipeline",
        "eogrow.pipelines.export_maps.ExportMapsPipeline",
    ],
)
def test_eogrow_template(command_params):
    command = f"eogrow-template {command_params}"
    assert subprocess.call(command, shell=True) == 0
