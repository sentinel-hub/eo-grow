"""Tasks used to transform Sentinel Hub Batch results into EOPatches."""

from __future__ import annotations

import concurrent.futures
import json

import fs
import numpy as np
from fs.base import FS

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import Feature
from eolearn.core.utils.fs import pickle_fs, unpickle_fs
from sentinelhub import parse_time

from ..utils.meta import import_object


class LoadUserDataTask(EOTask):
    """Task that loads and adds timestamps and tile names to an EOPatch"""

    def __init__(
        self,
        path: str,
        filesystem: FS,
        userdata_feature_name: str | None = None,
        userdata_timestamp_reader: str | None = None,
    ):
        """
        :param path: A path to folder containing the tiles, relative to the filesystem object.
        :param filesystem: A filesystem object.
        :param userdata_feature_name: A name of a META_INFO feature in which userdata.json content could be stored
        :param userdata_timestamp_reader: A reference to a Python function or a Python code that collects timestamps
            from loaded userdata.json
        """
        self.path = path
        self.pickled_filesystem = pickle_fs(filesystem)
        self.userdata_feature_name = userdata_feature_name
        self.userdata_timestamp_reader = userdata_timestamp_reader

    def _load_userdata_file(self, folder: str, filename: str = "userdata.json") -> dict:
        """Loads a content of a JSON file"""
        filesystem = unpickle_fs(self.pickled_filesystem)
        full_path = fs.path.join(self.path, folder, filename)

        userdata_text = filesystem.readtext(full_path, encoding="utf-8")
        return json.loads(userdata_text)

    @staticmethod
    def _parse_timestamps(userdata: dict, userdata_timestamp_reader: str) -> list:
        """Parses timestamps from userdata dictionary"""
        try:
            reader = import_object(userdata_timestamp_reader)
            time_strings = reader(userdata)
        except (ImportError, ValueError):
            time_strings = eval(userdata_timestamp_reader)  # pylint: disable=eval-used  # noqa: PGH001

        return [parse_time(time_string, force_datetime=True, ignoretz=True) for time_string in time_strings]

    def execute(self, eopatch: EOPatch | None = None, *, folder: str = "") -> EOPatch:
        """Adds metadata to the given EOPatch

        :param eopatch: Name of the eopatch to process
        :param folder: Folder in which userdata.json is stored
        """
        eopatch = eopatch or EOPatch()

        userdata = self._load_userdata_file(folder)

        if self.userdata_feature_name:
            eopatch.meta_info[self.userdata_feature_name] = userdata

        if self.userdata_timestamp_reader:
            eopatch.timestamps = self._parse_timestamps(userdata, self.userdata_timestamp_reader)

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

        if eopatch.timestamps:
            timeframe_num = len(eopatch.timestamps)
            if data.shape[0] != timeframe_num:  # Handling a case where data would contain some empty timeframes
                data = data[:timeframe_num, ...]

            order_mask = np.argsort(eopatch.timestamps)  # type: ignore[arg-type]
            is_strictly_increasing = (np.diff(order_mask) > 0).all()
            if not is_strictly_increasing:
                eopatch.timestamps = sorted(eopatch.timestamps)
                data = data[order_mask]

        eopatch[self.output_feature] = data
        return eopatch


class DeleteFilesTask(EOTask):
    """Delete files"""

    def __init__(self, path: str, filesystem: FS, filenames: list[str]):
        """
        :param path: A path to folder containing the files to be deleted, relative to filesystem object.
        :param filesystem: A filesystem object
        :param filenames: A list of filenames to delete
        """
        self.path = path
        self.pickled_filesystem = pickle_fs(filesystem)
        self.filenames = filenames

    def execute(self, *_: EOPatch, folder: str) -> None:
        """Execute method to delete files relative to the specified tile

        :param folder: A folder containing files
        """
        filesystem = unpickle_fs(self.pickled_filesystem)
        file_paths = [fs.path.join(self.path, folder, filename) for filename in self.filenames]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # The following is intentionally wrapped in a list in order to get back potential exceptions
            list(executor.map(filesystem.remove, file_paths))
