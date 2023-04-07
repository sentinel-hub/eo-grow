import os

import numpy as np
import pytest
from shapely.ops import unary_union

from sentinelhub import BBox

from eogrow.pipelines.import_tiff import ImportTiffPipeline
from eogrow.utils.testing import compare_content, generate_tiff_file, run_config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "experiment_name",
    [
        "import_tiff_temporal",
        "import_tiff_timeless",
        pytest.param("import_tiff_resized_new_size", marks=pytest.mark.chain),
        "import_tiff_resized_scale_factors",
    ],
)
def test_import_tiff_pipeline(config_and_stats_paths, experiment_name):
    config_path, stats_path = config_and_stats_paths("import_tiff", experiment_name)

    prepare_dummy_input(config_path)
    run_config(config_path)
    compare_content(config_path, stats_path)


def prepare_dummy_input(config_path: str) -> None:
    pipeline = ImportTiffPipeline.from_path(config_path)

    filesystem = pipeline.storage.filesystem
    input_folder = pipeline.storage.get_folder(pipeline.config.tiff_folder_key)
    input_file = os.path.join(input_folder, pipeline.config.input_filename)

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
