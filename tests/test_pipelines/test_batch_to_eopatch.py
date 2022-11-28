import json
import os
from typing import Iterable, Optional

import fs
import numpy as np
import pytest
from fs.base import FS

from sentinelhub import BBox

from eogrow.core.config import interpret_config_from_path
from eogrow.pipelines.batch_to_eopatch import BatchToEOPatchPipeline
from eogrow.utils.testing import ContentTester, check_pipeline_logs, create_folder_dict, generate_tiff_file


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "download_and_batch")


def prepare_batch_files(
    folder: str,
    filesystem: FS,
    filenames: Iterable[str],
    tiff_bbox: BBox,
    width: int,
    height: int,
    num_timestamps: int,
    dtype: type,
    add_userdata: bool,
    timestamp_shuffle_seed: Optional[int] = None,
) -> None:
    filesystem.makedirs(folder, recreate=True)
    generate_tiff_file(
        filesystem,
        (os.path.join(folder, file) for file in filenames),
        tiff_bbox=tiff_bbox,
        width=width,
        height=height,
        num_bands=num_timestamps,
        dtype=dtype,
    )

    if add_userdata:
        timestamp = [f"2020-11-{i}T01:23:45Z" for i in range(1, num_timestamps + 1)]
        if timestamp_shuffle_seed is not None:
            np.random.default_rng(timestamp_shuffle_seed).shuffle(timestamp)
        userdata = {"timestamps": timestamp}
        filesystem.writetext(fs.path.combine(folder, "userdata.json"), json.dumps(userdata))


@pytest.mark.parametrize(
    "experiment_name",
    [
        pytest.param("batch_to_eopatch", marks=pytest.mark.chain),
        "batch_to_eopatch_no_userdata",
    ],
)
def test_batch_to_eopatch_pipeline(folders, experiment_name):
    # Can't use utility testing due to custom pipeline
    config_filename = os.path.join(folders["config_folder"], experiment_name + ".json")
    stat_path = os.path.join(folders["stats_folder"], experiment_name + ".json")

    pipeline = BatchToEOPatchPipeline.from_raw_config(interpret_config_from_path(config_filename))

    filesystem = pipeline.storage.filesystem
    input_folder = pipeline.storage.get_folder(pipeline.config.input_folder_key)
    output_folder = pipeline.storage.get_folder(pipeline.config.output_folder_key)
    filesystem.removetree(input_folder)
    filesystem.removetree(output_folder)

    bboxes = pipeline.eopatch_manager.get_bboxes()
    folders = pipeline.eopatch_manager.get_eopatch_filenames()
    add_userdata = bool(pipeline.config.userdata_feature_name or pipeline.config.userdata_timestamp_reader)
    for bbox, patch_folder in zip(bboxes, folders):
        patch_path = fs.path.combine(input_folder, patch_folder)
        prepare_batch_files(
            folder=patch_path,
            filesystem=filesystem,
            filenames=["B01.tif", "B02.tif", "B03.tif"],
            tiff_bbox=bbox,
            width=200,
            height=300,
            num_timestamps=7,
            dtype=np.uint16,
            add_userdata=add_userdata,
            timestamp_shuffle_seed=17,
        )

    pipeline.run()
    check_pipeline_logs(pipeline)

    tester = ContentTester(pipeline.storage.filesystem, output_folder)
    # tester.save(stat_path)
    assert tester.compare(stat_path) == {}
