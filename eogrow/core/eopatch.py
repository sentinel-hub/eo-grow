"""
Module for handling EOPatch naming conventions
"""
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas
from bidict import bidict
from pandas import DataFrame
from pydantic import Field

from sentinelhub import CRS, BBox

from .area.base import AreaManager
from .area.batch import BatchAreaManager
from .base import EOGrowObject
from .schemas import ManagerSchema


class EOPatchManager(EOGrowObject):
    """Class defining and handling naming convention of EOPatches that divide AOI"""

    class Schema(ManagerSchema):
        pass

    def __init__(self, config: Schema, area_manager: AreaManager):
        """
        :param config: Configuration of the manager
        :param area_manager: Area managing class containing info about how the area is split into EOPatches
        """
        super().__init__(config)

        self._area_manager = area_manager

        self._name_to_id_map: Optional[bidict] = None
        self._name_to_bbox_map: Optional[dict] = None

    @property
    def name_to_id_map(self) -> bidict:
        """Provides a bidirectional mapping between EOPatch names and EOPatch ids. The mapping is generated lazily the
        first time it is needed.
        """
        if self._name_to_id_map is None:
            self._name_to_id_map, self._name_to_bbox_map = self._prepare_names()
        return self._name_to_id_map

    @property
    def name_to_bbox_map(self) -> dict:
        """Provides a dictionary mapping EOPatch names to bounding boxes."""
        if self._name_to_bbox_map is None:
            self._name_to_id_map, self._name_to_bbox_map = self._prepare_names()
        return self._name_to_bbox_map

    def _prepare_names(self) -> Tuple[bidict, Dict[str, BBox]]:
        """Collects bounding boxes from an instance of AreaManager class, generates EOPatch names and saves them into
        a private class variable _eopatch_name_list
        """
        bbox_grid = self._area_manager.get_grid(add_bbox_column=True)

        bbox_df: DataFrame = pandas.concat(bbox_grid, ignore_index=True)

        prepared_name_to_id_map = self.generate_names(bbox_df)
        prepared_name_to_bbox_map = dict(zip(prepared_name_to_id_map, bbox_df["BBOX"]))
        return prepared_name_to_id_map, prepared_name_to_bbox_map

    def generate_names(self, bbox_dataframe: DataFrame) -> bidict:
        """Method that generates EOPatch names from a dataframe holding bounding boxes. This method can be overridden
        by a method that generates names in a different way.

        :param bbox_dataframe: A dataframe with bounding box geometries and information
        :return: A bidirectional dictionary between names and original IDs
        """
        if len(bbox_dataframe) == 0:
            return bidict()

        total_patch_num = bbox_dataframe.total_num.values[0]
        zfill_length = len(str(total_patch_num - 1))

        return bidict(
            (f"eopatch-id-{row.index_n:0{zfill_length}}-col-{row.index_x}-row-{row.index_y}", int(row.index_n))
            for _, row in bbox_dataframe.iterrows()
        )

    def is_eopatch_name(self, name: str) -> bool:
        """Checks if the given name (could be entire file path) is the name of an EOPatch

        :param name: A name or a file path of a folder which could be one of EOPatches
        :type name: str
        """
        return os.path.basename(name) in self.name_to_id_map

    def get_eopatch_filenames(
        self, folder: Optional[str] = None, id_list: Optional[List[int]] = None, filter_existing: bool = True
    ) -> List[str]:
        """Provides a list of EOPatch names

        :param folder: If it is specified it will join the folder with eopatch names and return a list of complete paths
        :param id_list: A list of patch ids which should be provided. By default, is set to None and all EOPatch names
            will be provided
        :param filter_existing: If given a folder parameter this will check if EOPatches at the given location actually
            exist. By default, this is set to True which means it will make additional IO calls, which could be slow
            if using s3.
        :return: A list of EOPatch folder names
        """
        filenames = self._eopatch_list_from_id_list(id_list)

        if folder is None:
            return filenames

        file_paths = [os.path.join(folder, name) for name in filenames]

        if filter_existing:
            file_paths = [path for path in file_paths if os.path.isdir(path)]

            total_count, filtered_count = len(filenames), len(file_paths)
            if filtered_count < total_count:
                warnings.warn(
                    f"Only {filtered_count} out of {total_count} EOPatches found in folder {folder}", RuntimeWarning
                )
        return file_paths

    def save_eopatch_filenames(self, filename: str, eopatch_list: Union[List[int], List[str], None] = None) -> None:
        """Saves a list of EOPatches to a file

        :param filename: A filename (or entire file path) where names of EOPatches will be saved. Supported formats
            are JSON and TXT
        :param eopatch_list: Can either be a list of EOPatch names (or file paths) or a list of indices. If this is not
            specified, all names will be saved
        """
        if eopatch_list is None:
            eopatch_list = list(self.name_to_id_map)
        eopatch_list = self.parse_eopatch_list(eopatch_list)

        if filename.endswith(".json"):
            with open(filename, "w") as file:
                json.dump(eopatch_list, file, indent=2)
        elif filename.endswith(".txt"):
            with open(filename, "w") as file:
                print("\n".join(eopatch_list), end="", file=file)
        else:
            raise ValueError(f"Unrecognized file format of {filename}")

    def load_eopatch_filenames(self, filename: str, id_list: Optional[List[int]] = None) -> List[str]:
        """Loads a list of EOPatch names from a file

        :param filename: A filename (or entire file path) from where names of EOPatches will be loaded. Supported
            formats are JSON and TXT
        :param id_list: A list of EOPatch IDs which are the only ones required. The IDs are calculated according to
            the complete list of EOPatches. By default, no filtering is done.
        :return: A list of EOPatch names loaded from file (and maybe filtered)
        """
        if filename.endswith(".json"):
            with open(filename, "r") as file:
                eopatch_list = json.load(file)
        elif filename.endswith(".txt"):
            with open(filename, "r") as file:
                content = file.read()
                eopatch_list = [name.strip(" \t") for name in content.replace("\n", ",").split(",")]
                eopatch_list = [(int(name) if name.isdigit() else name) for name in eopatch_list if name]
        else:
            raise ValueError(f"Unrecognized file format of {filename}")

        eopatch_list = self.parse_eopatch_list(eopatch_list)

        if id_list is not None:
            filter_set = set(self._eopatch_list_from_id_list(id_list))
            eopatch_list = [name for name in eopatch_list if name in filter_set]

        return eopatch_list

    def get_bboxes(self, eopatch_list: Optional[List[str]] = None) -> List[BBox]:
        """Provides bounding boxes for each EOPatch

        :param eopatch_list:
        :return: A list of bounding box objects
        """
        if eopatch_list is None:
            return list(self.name_to_bbox_map.values())

        eopatch_list = self.parse_eopatch_list(eopatch_list)
        return [self.name_to_bbox_map[eopatch_name] for eopatch_name in eopatch_list]

    def parse_eopatch_list(self, eopatch_list: Union[List[str], List[int]]) -> List[str]:
        """Parses given list of EOPatches into a standard format

        :param eopatch_list: Can either be a list of EOPatch names (or file paths) or a list of indices
        """
        if not isinstance(eopatch_list, list):
            raise ValueError(f"Expected a list of EOPatch names, got {eopatch_list}")

        if all(isinstance(name, str) for name in eopatch_list):
            eopatch_list = cast(List[str], eopatch_list)  # informs mypy of the type, doesn't change anything
            eopatch_list = [os.path.basename(name) for name in eopatch_list]

            if set(eopatch_list) <= self.name_to_id_map.keys():
                return eopatch_list

            raise ValueError("The list of EOPatch names contains invalid names")

        if all(isinstance(name, int) for name in eopatch_list):
            return [self.name_to_id_map.inverse[idx] for idx in eopatch_list]

        raise ValueError(
            f"Elements of a list of EOPatch names should be all strings or all integers, got {eopatch_list}"
        )

    def _eopatch_list_from_id_list(self, id_list: Optional[List[int]]) -> List[str]:
        """A method that provides EOPatch list from id list"""
        if id_list is None:
            return list(self.name_to_id_map)

        return [self.name_to_id_map.inverse[eopatch_id] for eopatch_id in id_list]

    def get_id_list_from_eopatch_list(self, eopatch_list: Optional[List[str]] = None) -> List[int]:
        """Return patch ID given the eopatch name"""
        if eopatch_list is None:
            return list(self.name_to_id_map.values())
        return [self.name_to_id_map[eopatch] for eopatch in eopatch_list]

    def split_by_utm(self, eopatch_list: list) -> Dict[CRS, list]:
        """Creates dict of eopatches by utm."""
        crs_eopatch_dict: Dict[CRS, list] = {}
        for eopatch in eopatch_list:
            crs = self.name_to_bbox_map[eopatch].crs
            crs_eopatch_dict[crs] = crs_eopatch_dict.get(crs, [])
            crs_eopatch_dict[crs].append(eopatch)

        return crs_eopatch_dict


class CustomGridEOPatchManager(EOPatchManager):
    """A custom patch manager that uses a naming convention defined in a grid"""

    class Schema(EOPatchManager.Schema):
        name_column: str = Field(description="A name of a column in grid dataframes that contains EOPatch names")
        index_column: str = Field(description="A name of a column in grid dataframes that contains EOPatch indices")

    config: Schema

    def generate_names(self, bbox_dataframe: DataFrame, *_: Any, **__: Any) -> bidict:
        """Creates a bidirectional dictionary between names and indices"""
        names = bbox_dataframe[self.config.name_column]
        indices = bbox_dataframe[self.config.index_column]
        return bidict((name, index) for name, index in zip(names, indices))


class BatchTileManager(EOPatchManager):
    """A custom patch manager that uses a naming convention based on Sentinel Hub Batch tiles"""

    _area_manager: BatchAreaManager

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        if not isinstance(self._area_manager, BatchAreaManager):
            raise ValueError(f"{self.__class__.__name__} is only compatible with {BatchAreaManager.__name__}")

    def generate_names(self, bbox_dataframe: DataFrame, *_: Any, **__: Any) -> bidict:
        """Creates a bidirectional dictionary between names and indices"""
        if self._area_manager.subsplit != (1, 1):
            return bidict(
                (f"{row['name']}_{row.split_x}_{row.split_y}", row.index_n) for _, row in bbox_dataframe.iterrows()
            )

        return bidict((name, index) for name, index in zip(bbox_dataframe.name, bbox_dataframe.index_n))
