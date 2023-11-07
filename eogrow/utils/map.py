"""
Module with utilities for creating maps
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Literal

SH_COMMAND_LIMIT = 130000
OPEN_FILES_LIMIT = 1000
# ruff: noqa: RUF013

LOGGER = logging.getLogger(__name__)

GDAL_DTYPE_SETTINGS = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "float32": "Float32",
}
CogifyResamplingOptions = Literal[None, "NEAREST", "MODE", "AVERAGE", "BILINEAR", "CUBIC", "CUBICSPLINE", "LANCZOS"]
WarpResamplingOptions = Literal[
    None,
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "rms",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
]


def cogify_inplace(
    tiff_file: str,
    blocksize: int = 2048,
    nodata: float | None = None,
    dtype: Literal[None, "int8", "int16", "uint8", "uint16", "float32"] = None,
    resampling: CogifyResamplingOptions = None,
    quiet: bool = True,
) -> None:
    """Make the (geotiff) file a cog

    :param tiff_file: .tiff file to cogify
    :param blocksize: block size of tiled COG
    :param nodata: value to be treated as nodata, default value is None
    :param dtype: output type of the in the resulting tiff, default is None
    :param resampling: The resampling method used to produce overviews. The defaults (when using None) are CUBIC for
        floats and NEAREST for integers.
    :param quiet: The process does not produce logs.
    """
    temp_file = NamedTemporaryFile()
    temp_file.close()

    cogify(
        tiff_file,
        temp_file.name,
        blocksize,
        nodata=nodata,
        dtype=dtype,
        overwrite=True,
        resampling=resampling,
        quiet=quiet,
    )
    # Note: by moving the file we also remove the one at temp_file.name
    shutil.move(temp_file.name, tiff_file)


def cogify(
    input_file: str,
    output_file: str,
    blocksize: int = 1024,
    nodata: float | None = None,
    dtype: Literal[None, "int8", "int16", "uint8", "uint16", "float32"] = None,
    overwrite: bool = True,
    resampling: CogifyResamplingOptions = None,
    quiet: bool = True,
) -> None:
    """Create a cloud optimized version of input file

    :param input_file: File to cogify
    :param output_file: Resulting cog file
    :param blocksize: block size of tiled COG
    :param nodata: value to be treated as nodata, default value is None
    :param dtype: output type of the in the resulting tiff, default is None
    :param overwrite: If True overwrite the output file if it exists.
    :param resampling: The resampling method used to produce overviews. The defaults (when using None) are CUBIC for
        floats and NEAREST for integers.
    :param quiet: The process does not produce logs.
    """
    if input_file == output_file:
        raise OSError("Input file is the same as output file.")

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            raise OSError(f"{output_file} exists!")

    version = subprocess.check_output(("gdalinfo", "--version"), text=True).split(",")[0].replace("GDAL ", "")
    if version < "3.1.0":
        raise RuntimeError(
            f"The cogification process is configured for GDAL 3.1.0 and higher, but version {version} was detected.",
            RuntimeWarning,
        )

    gdaltranslate_options = (
        f"-of COG -co COMPRESS=DEFLATE -co BLOCKSIZE={blocksize} -co OVERVIEWS=IGNORE_EXISTING -co PREDICTOR=YES"
    )

    if resampling:
        gdaltranslate_options += f" -co RESAMPLING={resampling}"

    if quiet:
        gdaltranslate_options += " -q"

    if nodata is not None:
        gdaltranslate_options += f" -a_nodata {nodata}"

    if dtype is not None:
        gdaltranslate_options += f" -ot {GDAL_DTYPE_SETTINGS[dtype]}"

    if version < "3.6.0" and resampling == "MODE":
        warnings.warn(
            "GDAL versions below 3.6.0 have issues with `MODE` overview resampling. Trying to fix issue by setting"
            " GDAL_OVR_CHUNK_MAX_SIZE to a large integer (2100000000).",
            category=RuntimeWarning,
            stacklevel=2,
        )
        gdaltranslate_options += " --config GDAL_OVR_CHUNK_MAX_SIZE 2100000000"

    subprocess.check_call(f"gdal_translate {gdaltranslate_options} {input_file} {output_file}", shell=True)


def merge_tiffs(
    input_filenames: Iterable[str],
    merged_filename: str,
    *,
    overwrite: bool = True,
    nodata: float | None = None,
    dtype: Literal[None, "int8", "int16", "uint8", "uint16", "float32"] = None,
    warp_resampling: WarpResamplingOptions = None,
    quiet: bool = True,
) -> None:
    """Performs gdal_merge on a set of given geotiff images

    :param input_filenames: A sequence of input tiff image filenames
    :param merged_filename: Filename of merged tiff image
    :param overwrite: If True overwrite the output (merged) file if it exists
    :param delete_input: If True input images will be deleted at the end
    :param warp_resampling: The resampling method used when warping, useful for pixel misalignment. Defaults to NEAREST.
    :param quiet: The process does not produce logs.
    """
    gdalwarp_options = "-co BIGTIFF=YES -co compress=LZW -co TILED=YES"

    if overwrite:
        gdalwarp_options += " -overwrite"

    if quiet:
        gdalwarp_options += " -q"

    if warp_resampling:
        gdalwarp_options += f" -r {warp_resampling}"

    if nodata is not None:
        gdalwarp_options += f' -dstnodata "{nodata}"'  # B028

    if dtype is not None:
        gdalwarp_options += f" -ot {GDAL_DTYPE_SETTINGS[dtype]}"

    input_filelist = list(input_filenames)
    command = f"gdalwarp {gdalwarp_options} {' '.join(input_filelist)} {merged_filename}"

    if len(command) > SH_COMMAND_LIMIT or len(input_filelist) > OPEN_FILES_LIMIT:
        merged_path = Path(merged_filename)
        vrt_file_path = merged_path.with_name(f"{merged_path.stem}_temp.vrt")
        LOGGER.info("Command too big or too many files to process. Creating an intermediary vrt: %s", vrt_file_path)
        # generate text file with tile names & generate vrt
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as file_list:
            file_list.writelines([f"{tname}\n" for tname in input_filelist])
            generate_vrt_command = f"gdalbuildvrt {vrt_file_path} -input_file_list {file_list.name}"
            subprocess.check_call(generate_vrt_command, shell=True)

        # create merged file
        command = f"gdalwarp {gdalwarp_options} {vrt_file_path} {merged_filename}"
        subprocess.check_call(command, shell=True)

        # cleanup the vrt after the process
        os.remove(vrt_file_path)
    else:
        subprocess.check_call(command, shell=True)


def extract_bands(
    input_file: str,
    output_file: str,
    bands: Iterable[int],
    overwrite: bool = True,
    compress: bool = False,
    quiet: bool = True,
) -> None:
    """Extract bands from given input file

    :param input_file: File containing all bands
    :param output_file: Resulting file with extracted bands
    :param bands: Sequence of bands to extract. Indexation starts at 0.
    :param overwrite: If True overwrite the output file if it exists.
    :param quiet: The process does not produce logs.
    """
    if not bands:
        raise ValueError("No bands were specified for extraction, undefined behaviour.")

    if input_file == output_file:
        raise OSError("Input file is the same as output file.")

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            raise OSError(f"{output_file} already exists. Set `overwrite` to true if it should be overwritten.")

    # gdal_translate starts indexation at 1
    translate_opts = " ".join(f" -b {band + 1}" for band in bands)
    if quiet:
        translate_opts += " -q"
    if compress:
        translate_opts += " -co compress=LZW"

    command = f"gdal_translate {translate_opts} {input_file} {output_file}"
    subprocess.check_call(command, shell=True)
