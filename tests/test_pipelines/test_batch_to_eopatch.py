import json
import os
from typing import Iterable, Optional

import fs
import numpy as np
import pytest
from fs.base import FS

from sentinelhub import BBox

from eogrow.pipelines.batch_to_eopatch import BatchToEOPatchPipeline
from eogrow.utils.testing import compare_content, extract_output_folder, generate_tiff_file, run_config

pytestmark = pytest.mark.integration


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
    "experiment_name", [pytest.param("batch_to_eopatch", marks=pytest.mark.chain), "batch_to_eopatch_no_userdata"]
)
def test_batch_to_eopatch_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("download_and_batch", experiment_name)

    pipeline = BatchToEOPatchPipeline.from_path(config_path)

    filesystem = pipeline.storage.filesystem
    input_folder = pipeline.storage.get_folder(pipeline.config.input_folder_key)
    output_folder = pipeline.storage.get_folder(pipeline.config.output_folder_key)
    filesystem.removetree(input_folder)
    filesystem.removetree(output_folder)

    add_userdata = bool(pipeline.config.userdata_feature_name or pipeline.config.userdata_timestamp_reader)
    for patch_name, bbox in pipeline.get_patch_list():
        patch_path = fs.path.combine(input_folder, patch_name)
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

    run_config(config_path)
    compare_content(extract_output_folder(config_path), stats_path)
