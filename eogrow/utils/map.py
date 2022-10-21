"""
Module with utilities for creating maps
"""
import logging
import os
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Literal, Optional, Sequence

LOGGER = logging.getLogger(__name__)

GDAL_DTYPE_SETTINGS = {
    "uint8": "-ot Byte",
    "uint16": "-ot UInt16",
    "int16": "-ot Int16",
    "float32": "-ot Float32",
}


def cogify_inplace(
    tiff_file: str,
    blocksize: int = 2048,
    nodata: Optional[float] = None,
    dtype: Literal[None, "int8", "int16", "uint8", "uint16", "float32"] = None,
    quiet: bool = False,
) -> None:
    """Make the (geotiff) file a cog
    :param tiff_file: .tiff file to cogify
    :param blocksize: block size of tiled COG
    :param nodata: value to be treated as nodata, default value is None
    :param dtype: output type of the in the resulting tiff, default is None
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
        quiet=quiet,
    )
    shutil.move(temp_file.name, tiff_file)


def cogify(
    input_file: str,
    output_file: str,
    blocksize: int = 2048,
    nodata: Optional[float] = None,
    dtype: Literal[None, "int8", "int16", "uint8", "uint16", "float32"] = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Create a cloud optimized version of input file

    :param input_file: File to cogify
    :param output_file: Resulting cog file
    :param blocksize: block size of tiled COG
    :param nodata: value to be treated as nodata, default value is None
    :param dtype: output type of the in the resulting tiff, default is None
    :param overwrite: If True overwrite the output file if it exists.
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

    resampling = "AVERAGE" if dtype == "float32" else "NEAREST"

    gdaltranslate_options = (
        f"-of COG -co COMPRESS=DEFLATE -co BLOCKSIZE={blocksize} -co RESAMPLING={resampling} "
        "-co OVERVIEWS=IGNORE_EXISTING -co PREDICTOR=YES"
    )

    if quiet:
        gdaltranslate_options += " -q"

    if nodata is not None:
        gdaltranslate_options += f" -a_nodata {nodata}"

    if dtype is not None:
        gdaltranslate_options += f" {GDAL_DTYPE_SETTINGS[dtype]}"

    temp_filename = NamedTemporaryFile()
    temp_filename.close()
    shutil.copyfile(input_file, temp_filename.name)

    subprocess.check_call(f"gdal_translate {gdaltranslate_options} {temp_filename.name} {output_file}", shell=True)


def merge_tiffs(
    input_filename_list: List[str],
    merged_filename: str,
    *,
    overwrite: bool = False,
    nodata: Optional[float] = None,
    dtype: Literal[None, "int8", "int16", "uint8", "uint16", "float32"] = None,
    quiet: bool = False,
) -> None:
    """Performs gdal_merge on a set of given geotiff images

    :param input_filename_list: A list of input tiff image filenames
    :param merged_filename: Filename of merged tiff image
    :param overwrite: If True overwrite the output (merged) file if it exists
    :param delete_input: If True input images will be deleted at the end
    :param quiet: The process does not produce logs.
    """
    gdalwarp_options = "-co BIGTIFF=YES -co compress=LZW -co TILED=YES"

    if overwrite:
        gdalwarp_options += " -overwrite"
    if quiet:
        gdalwarp_options += " -q"

    if nodata is not None:
        gdalwarp_options += f' -dstnodata "{nodata}"'

    if dtype is not None:
        gdalwarp_options += f" {GDAL_DTYPE_SETTINGS[dtype]}"

    command = f"gdalwarp {gdalwarp_options} {' '.join(input_filename_list)} {merged_filename} "
    subprocess.check_call(command, shell=True)


def extract_bands(
    input_file: str,
    output_file: str,
    bands: Sequence[int],
    overwrite: bool = False,
    quiet: bool = False,
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
    translate_opts = "-co compress=LZW" + " ".join(f" -b {band + 1}" for band in bands)
    if quiet:
        translate_opts += " -q"

    temp_filename = NamedTemporaryFile()
    temp_filename.close()
    shutil.copyfile(input_file, temp_filename.name)

    subprocess.check_call(f"gdal_translate {translate_opts} {temp_filename.name} {output_file}", shell=True)
