"""
Testing batch-to-eopatch pipeline
"""
import json
import os

import fs
import numpy as np
import pytest
import rasterio
from fs.base import FS

from sentinelhub import BBox

from eogrow.core.config import Config
from eogrow.pipelines.batch_to_eopatch import BatchToEOPatchPipeline
from eogrow.utils.testing import ContentTester, check_pipeline_logs, create_folder_dict


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "download_and_batch")


def prepare_batch_files(
    folder: str,
    filesystem: FS,
    tiff_bbox: BBox,
    width: int,
    height: int,
    num_timestamps: int,
    dtype: str,
    add_userdata: bool,
):
    transform = rasterio.transform.from_bounds(*tiff_bbox, width=width, height=height)
    filesystem.makedirs(folder, recreate=True)

    generator = np.random.default_rng(42)
    for filename in ["B01.tif", "B02.tif", "B03.tif"]:
        with filesystem.openbin(fs.path.combine(folder, filename), "w") as file_handle:
            with rasterio.open(
                file_handle,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=num_timestamps,
                dtype=dtype,
                nodata=0,
                transform=transform,
                crs=tiff_bbox.crs.ogc_string(),
            ) as dst:

                dst.write(10000 * generator.random((num_timestamps, height, width)))

    if add_userdata:
        userdata = {"timestamps": [f"2020-11-{i}T01:23:45Z" for i in range(1, num_timestamps + 1)]}
        filesystem.writetext(fs.path.combine(folder, "userdata.json"), json.dumps(userdata))


@pytest.mark.chain
@pytest.mark.parametrize(
    "experiment_name",
    [
        "batch_to_eopatch",
        "batch_to_eopatch_no_userdata",
    ],
)
def test_rasterize_pipeline_preprocess(folders, experiment_name):
    # Can't use utility testing due to custom pipeline
    config_filename = os.path.join(folders["config_folder"], experiment_name + ".json")
    stat_path = os.path.join(folders["stats_folder"], experiment_name + ".json")

    pipeline = BatchToEOPatchPipeline(Config.from_path(config_filename))

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
        prepare_batch_files(patch_path, filesystem, bbox, 400, 800, 12, "uint16", add_userdata)

    pipeline.run()
    check_pipeline_logs(pipeline)

    tester = ContentTester(pipeline.storage.filesystem, output_folder)
    # tester.save(stat_path)
    assert tester.compare(stat_path) == {}
