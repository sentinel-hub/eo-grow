"""
Tests for utils.map module
"""
from typing import List, Optional

import numpy as np
import pytest
import rasterio
from fs.base import FS
from fs.tempfs import TempFS
from numpy.testing import assert_array_almost_equal
from pytest import approx

from eolearn.core import EOPatch, FeatureType
from eolearn.io import ExportToTiffTask
from sentinelhub import CRS, BBox

from eogrow.utils.map import GDAL_DTYPE_SETTINGS, cogify, cogify_inplace, extract_bands, merge_tiffs

pytestmark = pytest.mark.fast


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

    @pytest.fixture(name="output_path", scope="class")
    def output_file_fixture(_, filesystem: FS) -> str:
        return filesystem.getsyspath("output.tif")

    @pytest.mark.parametrize("dtype", GDAL_DTYPE_SETTINGS)
    @pytest.mark.parametrize("block", (1024, 64))
    @pytest.mark.parametrize("nodata", (None, 0, 11))
    def test_cogify(self, input_path: str, output_path: str, nodata: Optional[float], dtype: str, block: int) -> None:
        cogify(input_path, output_path, nodata=nodata, dtype=dtype, blocksize=block, overwrite=True)
        self._test_output_file(output_path, nodata, dtype, block)

    @pytest.mark.parametrize("dtype", ("float32", "uint8"))
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


class TestMerge:
    @pytest.fixture(name="output_path", scope="class")
    def output_file_fixture(_, filesystem: FS) -> str:
        return filesystem.getsyspath("output.tif")

    @pytest.fixture(name="input_paths", scope="class")
    def input_file_fixture(_, filesystem: FS) -> List[str]:
        for i, bbox_coords in enumerate([(9, 9, 10, 10), (10, 9, 11, 10), (9, 10, 10, 11)]):
            data = np.full((100, 100, 2), i, dtype="int16")
            make_geotiff(data, BBox(bbox_coords, CRS.WGS84), filesystem, name=f"input{i}")
        return [filesystem.getsyspath(f"input{i}.tif") for i in range(3)]

    def _expected_output(_, nodata: Optional[float]) -> np.ndarray:
        nodata = nodata or 0
        output = np.full((200, 200, 2), nodata)
        output[100:200, :100, ...] = 0
        output[100:200, 100:200, ...] = 1
        output[:100, :100, ...] = 2
        return output

    @pytest.mark.parametrize(
        "nodata, dtype",
        [
            (None, "float32"),
            (0, "float32"),
            (-3.7, "float32"),
            (-11, "int16"),
            (11, "uint16"),
            (2, "uint16"),
            (3, "uint8"),
        ],
    )
    def test_merge_tiffs(self, output_path: str, input_paths: List[str], nodata: Optional[float], dtype: str):
        merge_tiffs(input_paths, output_path, nodata=nodata, dtype=dtype, overwrite=True)

        with rasterio.open(output_path) as src:
            for tiff_dtype, tiff_nodata in zip(src.dtypes, src.nodatavals):
                assert tiff_dtype == dtype and tiff_nodata == approx(nodata)
            output = np.moveaxis(src.read(), 0, -1)
            assert_array_almost_equal(output, self._expected_output(nodata))


class TestExtractBands:
    @pytest.fixture(name="output_path", scope="class")
    def output_file_fixture(_, filesystem: FS) -> str:
        return filesystem.getsyspath("output.tif")

    @pytest.fixture(name="input_data", scope="class")
    def input_data_fixture(_) -> np.ndarray:
        layers = [np.zeros((100, 100)), np.ones((100, 100)), np.arange(100 * 100).reshape((100, 100))]
        return np.stack(layers, axis=-1).astype(np.int32)

    @pytest.fixture(name="input_path", scope="class")
    def input_file_fixture(_, input_data: np.ndarray, filesystem: FS) -> str:
        make_geotiff(input_data, BBox((1, 2, 3, 4), CRS.WGS84), filesystem, name="input")
        return filesystem.getsyspath("input.tif")

    @pytest.mark.parametrize("bands", [[0], [1], [2], [0, 1], [2, 1, 0]])
    def test_extract_bands(self, output_path: str, input_path: List[str], input_data: np.ndarray, bands: List[int]):
        extract_bands(input_path, output_path, bands, overwrite=True)

        with rasterio.open(output_path) as src:
            output = np.moveaxis(src.read(), 0, -1)
            assert_array_almost_equal(output, input_data[..., bands])

    def test_extract_bands_empty(self, output_path: str, input_path: List[str]):
        with pytest.raises(ValueError):
            extract_bands(input_path, output_path, [], overwrite=True)
