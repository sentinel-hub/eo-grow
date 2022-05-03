"""
Module implementing utilities for unit testing pipeline results
"""
import functools
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

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

    def __init__(self, filesystem: FS, main_folder: str, decimals: int = 5):
        """
        :param filesystem: A filesystem containing project data
        :param main_folder: A folder path on the filesystem where results are saved
        :param decimals: Number of decimals to which values will be rounded
        """
        self.filesystem = filesystem
        self.main_folder = main_folder
        self.decimals = decimals

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
        with open(filename, "w") as file:
            json.dump(self.stats, file, indent=2, sort_keys=True)

    def _calculate_stats(self, folder: Optional[str] = None) -> Dict[str, object]:
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
                stats[content] = self._calculate_numpy_stats(content_path)
            elif content_path.endswith(".aux.xml"):
                pass
            else:
                stats[content] = None

        return stats

    def _calculate_eopatch_stats(self, eopatch: EOPatch) -> Dict[str, object]:
        """Calculates statistics of given EOPatch and it's content"""
        stats: Dict[str, object] = {}

        for feature_type, feature_set in eopatch.get_features().items():
            feature_type_name = feature_type.value

            if feature_type is FeatureType.BBOX:
                stats[feature_type_name] = repr(eopatch.bbox)

            elif feature_type is FeatureType.TIMESTAMP:
                stats[feature_type_name] = [time.isoformat() for time in eopatch.timestamp]

            else:
                feature_stats_dict = {}

                if feature_type.is_raster():
                    calculation_method: Callable = self._calculate_raster_stats
                elif feature_type.is_vector():
                    calculation_method = self._calculate_vector_stats
                else:  # Only FeatureType.META_INFO remains
                    calculation_method = str

                for feature_name in feature_set:
                    feature_stats_dict[feature_name] = calculation_method(eopatch[feature_type][feature_name])

                stats[feature_type_name] = feature_stats_dict

        return stats

    def _calculate_raster_stats(self, raster: np.ndarray) -> List[object]:
        """Calculates statistics over a raster numpy array"""
        stats = [*raster.shape, str(raster.dtype)]
        if raster.dtype == object or raster.dtype.kind == "U":
            return stats

        stats.extend([np.count_nonzero(np.isnan(raster)), np.count_nonzero(np.isfinite(raster))])

        finite_values = raster[np.isfinite(raster)]

        if finite_values.size:
            if finite_values.dtype == bool:
                finite_values = finite_values.astype(np.uint8)

            for operation in [np.min, np.max, np.mean, np.median]:
                stats.append(round(float(operation(finite_values)), self.decimals))  # type: ignore

            stats.extend(map(int, np.histogram(finite_values, bins=8)[0]))

            np.random.seed(0)
            randomly_chosen_values = np.random.choice(finite_values, 8)
            stats.extend(map(lambda value: round(float(value), self.decimals), randomly_chosen_values))

        return stats

    def _calculate_tiff_stats(self, tiff_filename: str) -> List[object]:
        """Calculates statistics over a .tiff image"""
        with self.filesystem.openbin(tiff_filename, "r") as file_handle:
            with rasterio.open(file_handle) as tiff:
                raster = tiff.dataset_mask()

        return self._calculate_raster_stats(raster)

    def _calculate_numpy_stats(self, numpy_filename: str) -> List[object]:
        """Calculates statistics over a .npy file containing a numpy array"""
        with self.filesystem.openbin(numpy_filename, "r") as file_handle:
            raster = np.load(file_handle, allow_pickle=True)

        return self._calculate_raster_stats(raster)

    def _calculate_vector_stats(self, dataframe: pd.DataFrame) -> List[object]:
        """Calculates statistics over a vector GeoDataFrame"""  # TODO: add more statistical properties
        rounder = functools.partial(_round_point_coords, decimals=self.decimals)
        dataframe["geometry"] = dataframe["geometry"].apply(lambda geometry: shapely.ops.transform(rounder, geometry))

        stats = list(dataframe) + [len(dataframe.index), str(dataframe.crs)]

        if len(dataframe.index):
            stats.extend(list(map(str, dataframe.iloc[0])))

        return stats


def check_pipeline_logs(pipeline: Pipeline) -> None:
    """A utility function which checks pipeline logs and makes sure there are no failed executions"""
    if not pipeline.config.logging.save_logs:
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
            raise AssertionError(f"Expected and obtained stats differ: {stats_difference}")


def _round_point_coords(x: float, y: float, decimals: int) -> Tuple[float, float]:
    """Rounds coordinates of a point"""
    return round(x, decimals), round(y, decimals)


def create_folder_dict(config_folder: str, stats_folder: str, subfolder: Optional[str] = None) -> Dict[str, str]:
    return {
        "config_folder": os.path.join(config_folder, subfolder) if subfolder else config_folder,
        "stats_folder": os.path.join(stats_folder, subfolder) if subfolder else config_folder,
    }
