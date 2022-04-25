"""
Tasks used to transform Sentinel Hub Batch results into EOPatches
"""
import concurrent.futures
import json
from typing import List, Optional

import fs
import numpy as np

from eolearn.core import EOPatch, EOTask
from eolearn.core.utils.fs import get_base_filesystem_and_path
from sentinelhub import SHConfig, parse_time

from ..utils.meta import import_object
from ..utils.types import Feature


class LoadUserDataTask(EOTask):
    """Task that loads and adds timestamps and tile names to an EOPatch"""

    def __init__(
        self,
        path: str,
        userdata_feature_name: Optional[str] = None,
        userdata_timestamp_reader: Optional[str] = None,
        config: Optional[SHConfig] = None,
    ):
        """
        :param path: Path to folder containing the tiles
        :param userdata_feature_name: A name of a META_INFO feature in which userdata.json content could be stored
        :param userdata_timestamp_reader: A reference to a Python function or a Python code that collects timestamps
            from loaded userdata.json
        :param config: A configuration object with AWS credentials
        """
        self.path = path
        self.userdata_feature_name = userdata_feature_name
        self.userdata_timestamp_reader = userdata_timestamp_reader
        self.config = config

    def _load_userdata_file(self, folder: str, filename: str = "userdata.json") -> dict:
        """Loads a content of a JSON file"""
        filesystem, relative_path = get_base_filesystem_and_path(self.path, config=self.config)
        full_path = fs.path.join(relative_path, folder, filename)

        userdata_text = filesystem.readtext(full_path, encoding="utf-8")
        return json.loads(userdata_text)

    @staticmethod
    def _parse_timestamps(userdata: dict, userdata_timestamp_reader: str) -> list:
        """Parses timestamps from userdata dictionary"""
        try:
            reader = import_object(userdata_timestamp_reader)
            time_strings = reader(userdata)
        except (ImportError, ValueError):
            time_strings = eval(userdata_timestamp_reader)  # pylint: disable=eval-used

        return [parse_time(time_string, force_datetime=True, ignoretz=True) for time_string in time_strings]

    def execute(self, eopatch: Optional[EOPatch] = None, *, folder: str = "") -> EOPatch:
        """Adds metadata to the given EOPatch

        :param eopatch: Name of the eopatch to process
        :param folder: Folder in which userdata.json is stored
        """
        eopatch = eopatch or EOPatch()

        userdata = self._load_userdata_file(folder)

        if self.userdata_feature_name:
            eopatch.meta_info[self.userdata_feature_name] = userdata

        if self.userdata_timestamp_reader:
            eopatch.timestamp = self._parse_timestamps(userdata, self.userdata_timestamp_reader)

        return eopatch


class FixImportedTimeDependentFeatureTask(EOTask):
    """Fixes a time-dependent feature that has been imported as a timeless feature from batch results.

    It performs the following:

    -  rotates bands axis to time axis,
    -  reverses order of timestamps and feature
    -  potentially removes redundant timeframes according to timestamps. This is necessary because in case there were
       no available acquisitions batch job still had to save a single time frame with dummy values.

    """

    def __init__(self, input_feature: Feature, output_feature: Feature):
        self.input_feature = input_feature
        self.output_feature = output_feature

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Fixes a feature in the given EOPatch"""
        data = eopatch[self.input_feature]
        del eopatch[self.input_feature]

        data = data[np.newaxis, ...]
        data = np.swapaxes(data, 0, -1)

        if eopatch.timestamp:
            timeframe_num = len(eopatch.timestamp)
            if data.shape[0] != timeframe_num:  # Handling a case where data would contain some empty timeframes
                data = data[:timeframe_num, ...]

            order_mask = np.argsort(eopatch.timestamp)
            is_strictly_increasing = (np.diff(order_mask) > 0).all()
            if not is_strictly_increasing:
                eopatch.timestamp = sorted(eopatch.timestamp)
                data = data[order_mask]

        eopatch[self.output_feature] = data
        return eopatch


class DeleteFilesTask(EOTask):
    """Delete files"""

    def __init__(self, path: str, filenames: List[str], config: Optional[SHConfig] = None):
        """
        :param path: Path to folder containing the files to be deleted
        :type path: str
        :param filenames: A list of filenames to delete
        :type filenames: list(str)
        :param config: A configuration object with AWS credentials
        :type config: SHConfig
        """
        self.path = path
        self.filenames = filenames
        self.config = config

    def execute(self, *_: EOPatch, folder: str) -> None:
        """Execute method to delete files relative to the specified tile

        :param folder: A folder containing files
        """
        filesystem, relative_path = get_base_filesystem_and_path(self.path, config=self.config)

        file_paths = [fs.path.join(relative_path, folder, filename) for filename in self.filenames]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # The following is intentionally wrapped in a list in order to get back potential exceptions
            list(executor.map(filesystem.remove, file_paths))
