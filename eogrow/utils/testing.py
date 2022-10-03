"""
Module implementing utilities for unit testing pipeline results
"""
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import fs
import numpy as np
import pandas as pd
import rasterio
import shapely.ops
from deepdiff import DeepDiff
from fs.base import FS

from eolearn.core import EOPatch, FeatureType

from ..core.config import collect_configs_from_path, interpret_config_from_dict
from ..core.pipeline import Pipeline
from ..utils.meta import load_pipeline_class
from ..utils.types import JsonDict


class ContentTester:
    """Utility used for testing pipeline results

    Results of a pipeline is usually a folder containing multiple EOPatches, each containing various features. This
    utility aggregates all folder content into some basic statistics.

    * Every time you initialize this class statistics will be calculated
    * Statistics can be saved into a JSON file (it's human-readable)
    * Statistics can be compared with the one saved in a file from the previous run.

    If statistics match there is a good chance that the pipeline produced exactly the same results as before. Otherwise,
    this utility will let you know which statistics does not match
    """

    _STATS_OPERATIONS: Dict[str, Callable] = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "median": np.median,
    }

    def __init__(
        self,
        filesystem: FS,
        main_folder: str,
        decimals: int = 5,
        unique_values_limit: int = 8,
        histogram_bin_num: int = 8,
        max_random_values: int = 5,
    ):
        """
        :param filesystem: A filesystem containing project data
        :param main_folder: A folder path on the filesystem where results are saved
        :param decimals: Number of decimals to which values will be rounded
        :param unique_values_limit: If a raster has at most this many unique values then statistics will show all
            unique values with their counts. Otherwise, multiple statistical properties will be calculated for
            the values.
        :param histogram_bin_num: Number of bins in a histogram for statistics. The histogram will be calculated only
            if number of unique values is higher than `unique_values_limit`.
        :param max_random_values: Number of values that will be randomly sampled from an array for statistics. This
            will happen only if the array contains at least `2` different unique values.
        """
        self.filesystem = filesystem
        self.main_folder = main_folder
        self.decimals = decimals
        self.unique_values_limit = unique_values_limit
        self.histogram_bin_num = histogram_bin_num
        self.max_random_values = max_random_values

        if not filesystem.isdir(self.main_folder):
            raise ValueError(f"Folder {self.main_folder} does not exist on filesystem {self.filesystem}")

        self.stats = self._calculate_stats()

    def compare(self, filename: str) -> DeepDiff:
        """Compares statistics of given folder content with statistics saved in a given file

        :param filename: A JSON filename (with file path) where expected statistics is saved
        :return: A dictionary report about differences between expected and actual content
        """
        with open(filename, "r") as file:
            expected_stats = json.load(file)

        return DeepDiff(expected_stats, self.stats)

    def save(self, filename: str) -> None:
        """Saves statistics of given folder content into a JSON file

        :param filename: A JSON filename (with file path) where statistics should be saved
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            json.dump(self.stats, file, indent=2, sort_keys=True)

    def _calculate_stats(self, folder: Optional[str] = None) -> JsonDict:
        """Calculates statistics of given folder and it's content"""
        stats: Dict[str, object] = {}
        if folder is None:
            folder = self.main_folder

        for content in self.filesystem.listdir(folder):
            content_path = fs.path.combine(folder, content)

            if self.filesystem.isdir(content_path):
                eopatch = EOPatch.load(content_path, filesystem=self.filesystem)

                if eopatch.bbox:
                    stats[content] = self._calculate_eopatch_stats(eopatch)
                else:  # Probably it is not an EOPatch folder
                    stats[content] = self._calculate_stats(folder=content_path)
            elif content_path.endswith("tiff"):
                stats[content] = self._calculate_tiff_stats(content_path)
            elif content_path.endswith(".npy"):
                stats[content] = self._calculate_numpy_file_stats(content_path)
            elif content_path.endswith(".aux.xml"):
                pass
            else:
                stats[content] = None

        return stats

    def _calculate_eopatch_stats(self, eopatch: EOPatch) -> JsonDict:
        """Calculates statistics of given EOPatch and it's content"""
        stats: Dict[str, object] = {}

        for feature_type, feature_set in eopatch.get_features().items():
            feature_type_name = feature_type.value

            if feature_type is FeatureType.BBOX:
                stats[feature_type_name] = repr(eopatch.bbox)

            elif feature_type is FeatureType.TIMESTAMP:
                stats[feature_type_name] = [time.isoformat() for time in eopatch.timestamp]

            else:
                feature_set = cast(Set[str], feature_set)
                feature_stats_dict = {}

                if feature_type.is_raster():
                    calculation_method: Callable = self._calculate_numpy_stats
                elif feature_type.is_vector():
                    calculation_method = self._calculate_vector_stats
                else:  # Only FeatureType.META_INFO remains
                    calculation_method = str

                for feature_name in feature_set:
                    feature_stats_dict[feature_name] = calculation_method(eopatch[feature_type][feature_name])

                stats[feature_type_name] = feature_stats_dict

        return stats

    def _calculate_numpy_stats(self, raster: np.ndarray) -> JsonDict:
        """Calculates statistics over a raster numpy array"""
        stats: JsonDict = {"array_shape": list(raster.shape), "dtype": str(raster.dtype)}
        if raster.dtype == object or raster.dtype.kind == "U":
            return stats

        unique_values = np.unique(raster)

        if unique_values.size <= self.unique_values_limit:
            values, counts = np.unique(raster, return_counts=True)
            stats["values"] = [
                {"value": self._prepare_value(value), "count": int(count)} for value, count in zip(values, counts)
            ]

        else:
            number_values = raster[~np.isnan(raster)]
            finite_values = number_values[np.isfinite(number_values)]

            stats["counts"] = {
                "nan": raster.size - number_values.size,
                "infinite": number_values.size - finite_values.size,
            }
            stats["basic_stats"] = {
                name: self._prepare_value(operation(finite_values))
                for name, operation in self._STATS_OPERATIONS.items()
            }

            stats["subsample_basic_stats"] = self._calculate_subsample_stats(finite_values)

            counts, edges = np.histogram(finite_values, bins=self.histogram_bin_num)
            stats["histogram"] = {
                "counts": counts.astype(int).tolist(),
                "edges": list(map(self._prepare_value, edges)),
            }

        if unique_values.size > 1:
            stats["random_values"] = self._get_random_stats(raster, unique_values)

        return stats

    def _calculate_tiff_stats(self, tiff_filename: str) -> JsonDict:
        """Calculates statistics over a .tiff image"""
        with self.filesystem.openbin(tiff_filename, "r") as file_handle:
            with rasterio.open(file_handle) as tiff:
                image = tiff.read()
                mask = tiff.dataset_mask()

        return {
            "image": self._calculate_numpy_stats(image),
            "mask": self._calculate_numpy_stats(mask),
        }

    def _calculate_numpy_file_stats(self, numpy_filename: str) -> JsonDict:
        """Calculates statistics over a .npy file containing a numpy array"""
        with self.filesystem.openbin(numpy_filename, "r") as file_handle:
            raster = np.load(file_handle, allow_pickle=True)

        return self._calculate_numpy_stats(raster)

    def _calculate_vector_stats(self, dataframe: pd.DataFrame) -> JsonDict:
        """Calculates statistics over a vector GeoDataFrame"""  # TODO: add more statistical properties
        rounder = functools.partial(_round_point_coords, decimals=self.decimals)
        dataframe["geometry"] = dataframe["geometry"].apply(lambda geometry: shapely.ops.transform(rounder, geometry))

        stats = {
            "columns": list(dataframe),
            "row_count": len(dataframe.index),
            "crs": str(dataframe.crs),
        }

        if len(dataframe.index):
            stats["first_row"] = list(map(str, dataframe.iloc[0]))

        return stats

    def _calculate_subsample_stats(self, values: np.ndarray, amount: float = 0.1) -> Dict[str, float]:
        """Randomly samples a small amount of points from the array (10% by default) to recalculate the statistics.
        This introduces a 'positional instability' so that accidental mirroring or re-orderings are detected."""
        rng = np.random.default_rng(0)
        subsample = rng.choice(values, int(values.size * amount))
        return {name: self._prepare_value(operation(subsample)) for name, operation in self._STATS_OPERATIONS.items()}

    def _get_random_stats(self, raster: np.ndarray, unique_values: np.ndarray) -> List[JsonDict]:
        """First it randomly samples a few values from the list of unique values. Then for each one it checks where
        this value is located in the original array and randomly selects one of its locations. Selected locations
        and values are used for statistics."""
        rng = np.random.default_rng(0)
        randomly_chosen_values = rng.choice(
            unique_values,
            size=min(self.max_random_values, unique_values.size),
            replace=False,
        )

        random_stats: List[JsonDict] = []
        for value in randomly_chosen_values:
            value_mask = np.isnan(raster) if np.isnan(value) else raster == value
            positions = np.argwhere(value_mask)

            num_positions = positions.shape[0]
            chosen_position_index = rng.integers(num_positions)

            random_stats.append(
                {"value": self._prepare_value(value), "position": positions[chosen_position_index].tolist()}
            )

        return random_stats

    def _prepare_value(self, value: Any) -> Any:
        """Converts a value in a way that it can be compared and serialized into a JSON. It also rounds float values."""
        if not np.isscalar(value):
            return value
        if not np.isfinite(value):
            return repr(value)
        if np.issubdtype(type(value), np.integer):
            value = cast(int, value)
            return int(value)
        if np.issubdtype(type(value), bool):
            return bool(value)
        value = cast(float, value)
        return round(float(value), self.decimals)


def check_pipeline_logs(pipeline: Pipeline) -> None:
    """A utility function which checks pipeline logs and makes sure there are no failed executions"""
    if not pipeline.logging_manager.config.save_logs:
        raise ValueError("Pipeline did not save logs, this test would be useless")

    logs_folder = pipeline.logging_manager.get_pipeline_logs_folder(pipeline.current_execution_name)

    for filename in ["failed.json", "finished.json", "pipeline-report.json", "pipeline.log"]:
        path = fs.path.combine(logs_folder, filename)
        assert pipeline.storage.filesystem.isfile(path), f"File {path} is missing"

    logs_folder = pipeline.logging_manager.get_pipeline_logs_folder(pipeline.current_execution_name, full_path=True)
    failed_filename = fs.path.combine(logs_folder, "failed.json")
    assert not pipeline.eopatch_manager.load_eopatch_filenames(
        failed_filename
    ), f"Some executions failed, check {logs_folder}"

    finished_filename = os.path.join(logs_folder, "finished.json")
    assert pipeline.eopatch_manager.load_eopatch_filenames(finished_filename), "No executions finished"


def run_and_test_pipeline(
    experiment_name: str,
    *,
    config_folder: str,
    stats_folder: str,
    folder_key: Optional[str] = None,
    reset_folder: bool = True,
    save_new_stats: bool = False,
) -> None:
    """A default way of testing a pipeline

    :param experiment_name: Name of test experiment, which defines its config and stats filenames
    :param config_folder: A path to folder containing the config file
    :param stats_folder: A path to folder containing the file with expected result stats
    :param folder_key: Type of the folder containing results of the pipeline, if missing it's inferred from config
    :param reset_folder: If True it will delete content of the folder with results before running the pipeline
    :param save_new_stats: If True then new file with expected result stats will be saved, potentially overwriting the
        old one. Otherwise, the old one will be used to compare stats.
    """
    config_filename = os.path.join(config_folder, experiment_name + ".json")
    expected_stats_file = os.path.join(stats_folder, experiment_name + ".json")

    crude_configs = collect_configs_from_path(config_filename)
    raw_configs = [interpret_config_from_dict(config) for config in crude_configs]

    for index, config in enumerate(raw_configs):
        output_folder_key = folder_key or config.get("output_folder_key")
        if output_folder_key is None:
            raise ValueError(
                "Pipeline does not have a `output_folder_key` parameter, `folder_key` must be set by hand."
            )

        pipeline = load_pipeline_class(config).from_raw_config(config)

        folder = pipeline.storage.get_folder(output_folder_key)
        filesystem = pipeline.storage.filesystem

        if reset_folder:
            filesystem.removetree(folder)
        pipeline.run()

        check_pipeline_logs(pipeline)

        if index < len(raw_configs) - 1:
            continue

        tester = ContentTester(filesystem, folder)

        if save_new_stats:
            tester.save(expected_stats_file)

        stats_difference = tester.compare(expected_stats_file)
        if stats_difference:
            stats_difference_repr = stats_difference.to_json(indent=2, sort_keys=True)
            raise AssertionError(f"Expected and obtained stats differ:\n{stats_difference_repr}")


def _round_point_coords(x: float, y: float, decimals: int) -> Tuple[float, float]:
    """Rounds coordinates of a point"""
    return round(x, decimals), round(y, decimals)


def create_folder_dict(config_folder: str, stats_folder: str, subfolder: Optional[str] = None) -> Dict[str, str]:
    return {
        "config_folder": os.path.join(config_folder, subfolder) if subfolder else config_folder,
        "stats_folder": os.path.join(stats_folder, subfolder) if subfolder else config_folder,
    }
