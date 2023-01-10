import os

import numpy as np
import pytest
from shapely.ops import unary_union

from sentinelhub import BBox

from eogrow.core.config import interpret_config_from_path
from eogrow.pipelines.import_tiff import ImportTiffPipeline
from eogrow.utils.testing import ContentTester, check_pipeline_logs, create_folder_dict, generate_tiff_file


@pytest.fixture(scope="session", name="folders")
def config_folder_fixture(config_folder, stats_folder):
    return create_folder_dict(config_folder, stats_folder, "import_tiff")


@pytest.mark.parametrize(
    "experiment_name",
    [
        "import_tiff_temporal",
        "import_tiff_timeless",
        pytest.param("import_tiff_resized_new_size", marks=pytest.mark.chain),
        "import_tiff_resized_scale_factors",
    ],
)
def test_import_tiff_pipeline(folders, experiment_name):
    # Can't use utility testing due to custom pipeline
    config_filename = os.path.join(folders["config_folder"], f"{experiment_name}.json")
    stat_path = os.path.join(folders["stats_folder"], f"{experiment_name}.json")

    pipeline = ImportTiffPipeline.from_raw_config(interpret_config_from_path(config_filename))

    filesystem = pipeline.storage.filesystem
    input_folder = pipeline.storage.get_folder(pipeline.config.tiff_folder_key)
    input_file = os.path.join(input_folder, pipeline.config.input_filename)

    output_folder = pipeline.storage.get_folder(pipeline.config.output_folder_key)
    filesystem.removetree(output_folder)

    bboxes = [bbox for _, bbox in pipeline.get_patch_list()]
    # one EOPatch gets left out so that we have some 'missing area', we also buffer it so that we have 'extra area'
    tiff_geom = unary_union([bbox.geometry for bbox in bboxes[:-1]]).buffer(100)

    generate_tiff_file(
        filesystem,
        [input_file],
        tiff_bbox=BBox(tiff_geom.bounds, crs=bboxes[0].crs),
        width=1420,  # magic numbers reverse engineered from area config
        height=705,
        num_bands=1 if pipeline.config.output_feature[0].is_timeless() else 3,
        dtype=np.float32,
    )

    pipeline.run()
    check_pipeline_logs(pipeline)

    tester = ContentTester(pipeline.storage.filesystem, output_folder)
    # tester.save(stat_path)
    assert tester.compare(stat_path) == {}
