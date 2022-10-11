"""
Tests for utils.map module
"""
from typing import Optional

import numpy as np
import pytest
import rasterio
from fs.base import FS
from fs.tempfs import TempFS

from eolearn.core import EOPatch, FeatureType
from eolearn.io import ExportToTiffTask
from sentinelhub import CRS, BBox

from eogrow.utils.map import GDAL_DTYPE_SETTINGS, cogify, cogify_inplace


def make_geotiff(data: np.ndarray, bbox: BBox, filesystem: FS, *, folder: str = ".", name: str):
    feature = (FeatureType.DATA_TIMELESS, "data")
    eopatch = EOPatch(bbox=bbox)
    eopatch[feature] = data
    ExportToTiffTask(feature, folder, filesystem=filesystem).execute(eopatch, filename=name)


@pytest.fixture(name="filesystem", scope="class")
def filesystem_fixture():
    with TempFS("eogrow_test_map_tempfs") as fs:
        yield fs


class TestCogify:
    @pytest.fixture(name="input_path")
    def input_file_fixture(_, filesystem: FS) -> str:
        # this is not class scoped because of in-place tests, and also it's lightning-speed compared to cogification
        data = np.arange(100 * 100 * 2).reshape((100, 100, 2)).astype(np.float32)
        make_geotiff(data, BBox((1, 2, 3, 4), CRS.WGS84), filesystem, name="input")
        return filesystem.getsyspath("input.tif")

    @pytest.fixture(name="output_path")
    def output_file_fixture(_, filesystem: FS) -> str:
        return filesystem.getsyspath("output.tif")

    @pytest.mark.parametrize("dtype", GDAL_DTYPE_SETTINGS)
    @pytest.mark.parametrize("block", (1024, 64))
    @pytest.mark.parametrize("nodata", (None, 0, 11))
    def test_cogify(self, input_path: str, output_path: str, nodata: Optional[float], dtype: str, block: int) -> None:
        cogify(input_path, output_path, nodata=nodata, dtype=dtype, blocksize=block, overwrite=True)
        self._test_output_file(output_path, nodata, dtype, block)

    @pytest.mark.parametrize("dtype", ("float32", "int8"))
    @pytest.mark.parametrize("block", (1024, 64))
    @pytest.mark.parametrize("nodata", (None, 11))
    def test_cogify_inplace(self, input_path: str, nodata: Optional[float], dtype: str, block: int) -> None:
        cogify_inplace(input_path, nodata=nodata, dtype=dtype, blocksize=block)
        self._test_output_file(input_path, nodata, dtype, block)

    @staticmethod
    def _test_output_file(path: str, nodata: Optional[float], dtype: str, blocksize: int) -> None:
        with rasterio.open(path) as src:
            assert src.block_shapes[0] == (blocksize, blocksize)
            for tiff_dtype, tiff_nodata in zip(src.dtypes, src.nodatavals):
                assert tiff_dtype == dtype and tiff_nodata == nodata
                assert len(set(src.block_shapes)) == 1, "tiff has multiple blocksizes"
