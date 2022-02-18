"""
Module with utilities for creating maps
"""
import logging
import os
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Union

LOGGER = logging.getLogger(__name__)


def cogify_inplace(tiff_file: str, blocksize: int = 2048, nodata: Union[None, int, float] = None) -> None:
    """Make the (geotiff) file a cog
    :param tiff_file: .tiff file to cogify
    :param blocksize: block size of tiled COG
    :param nodata: value to be treated as nodata, default value is None
    """
    temp_file = NamedTemporaryFile()
    temp_file.close()

    cogify(tiff_file, temp_file.name, blocksize, nodata=nodata, overwrite=True)
    shutil.move(temp_file.name, tiff_file)


def cogify(
    input_file: str,
    output_file: str,
    blocksize: int = 2048,
    nodata: Union[None, int, float] = None,
    overwrite: bool = False,
) -> None:
    """Create a cloud optimized version of input file

    :param input_file: File to cogify
    :param output_file: Resulting cog file
    :param blocksize: block size of tiled COG
    :param nodata: value to be treated as nodata, default value is None
    :param overwrite: If True overwrite the output file if it exists.
    """
    if input_file == output_file:
        raise OSError("Input file is the same as output file.")

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            raise OSError(f"{output_file} exists!")

    gdaladdo_options = f"-r mode --config GDAL_TIFF_OVR_BLOCKSIZE {blocksize} 2 4 8 16 32"

    gdaltranslate_options = (
        "-co TILED=YES -co COPY_SRC_OVERVIEWS=YES "
        f"--config GDAL_TIFF_OVR_BLOCKSIZE {blocksize} -co BLOCKXSIZE={blocksize} "
        f"-co BLOCKYSIZE={blocksize} -co COMPRESS=DEFLATE"
    )

    if nodata is not None:
        gdaltranslate_options += f" -a_nodata {nodata}"

    temp_filename = NamedTemporaryFile()
    temp_filename.close()
    shutil.copyfile(input_file, temp_filename.name)

    LOGGER.info("cogifying %s", input_file)
    subprocess.check_call(f"gdaladdo {temp_filename.name} {gdaladdo_options}", shell=True)
    subprocess.check_call(f"gdal_translate {gdaltranslate_options} {temp_filename.name} {output_file}", shell=True)
    LOGGER.info("cogifying done")


def merge_maps(
    input_filename_list: List[str],
    merged_filename: str,
    *,
    blocksize: int = 2048,
    nodata: Union[None, int, float] = None,
    cogify: bool = False,
    delete_input: bool = False,
) -> None:
    """Performs gdal_merge on a set of given geotiff images, make them pixel aligned cogs if `cogify` is True

    :param input_filename_list: A list of input tiff image filenames
    :param merged_filename: Filename of merged tiff image
    :param blocksize: block size of tiled COG
    :param nodata: which values should be treated as nodata in resulting COG, default is None
    :param cogify: If True make the final geotiff a COG.
    :param delete_input: If True input images will be deleted at the end
    """

    merge_tiffs(input_filename_list, merged_filename, overwrite=True, delete_input=delete_input)

    if cogify:
        cogify_inplace(merged_filename, blocksize, nodata=nodata)


def merge_tiffs(
    input_filename_list: List[str], merged_filename: str, *, overwrite: bool = False, delete_input: bool = False
) -> None:
    """Performs gdal_merge on a set of given geotiff images

    :param input_filename_list: A list of input tiff image filenames
    :param merged_filename: Filename of merged tiff image
    :param overwrite: If True overwrite the output (merged) file if it exists
    :param delete_input: If True input images will be deleted at the end
    """
    if os.path.exists(merged_filename):
        if overwrite:
            os.remove(merged_filename)
        else:
            raise OSError(f"{merged_filename} exists!")

    LOGGER.info("merging %d tiffs to %s", len(input_filename_list), merged_filename)
    subprocess.check_call(
        ["gdal_merge.py", "-co", "BIGTIFF=YES", "-co", "compress=LZW", "-o", merged_filename, *input_filename_list]
    )
    LOGGER.info("merging done")

    if delete_input:
        LOGGER.info("deleting input files")
        for filename in input_filename_list:
            if os.path.isfile(filename):
                os.remove(filename)
