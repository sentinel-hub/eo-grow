"""
Module implementing utilities for unit testing pipeline results
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, cast

import fs
import geopandas as gpd
import numpy as np
import rasterio
from deepdiff import DeepDiff
from fs.base import FS
from fs.osfs import OSFS
from shapely import MultiPolygon, Point, Polygon

from eolearn.core import EOPatch, FeatureType
from eolearn.core.eodata_io import get_filesystem_data_info
from sentinelhub import BBox

from ..core.config import collect_configs_from_path, interpret_config_from_dict
from ..core.pipeline import Pipeline
from ..types import JsonDict
from ..utils.eopatch_list import load_names
from ..utils.general import jsonify
from ..utils.meta import load_pipeline_class


@dataclass(frozen=True)
class StatCalcConfig:
    unique_values_limit: int = 8
    histogram_bin_num: int = 8
    num_random_values: int = 8


def compare_with_saved(stats: JsonDict, filename: str) -> DeepDiff:
    """Compares statistics of given folder content with statistics saved in a given file

    :param stats: Dictionary of calculated statistics of content
    :param filename: A JSON filename (with file path) where expected statistics is saved
    :return: A dictionary report about differences between expected and actual content
    """
    with open(filename) as file:
        expected_stats = json.load(file)

    return DeepDiff(expected_stats, stats)


def save_statistics(stats: JsonDict, filename: str) -> None:
    """Saves statistics of given folder content into a JSON file

    :param stats: Dictionary of calculated statistics of content
    :param filename: A JSON filename (with file path) where statistics should be saved
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(stats, file, indent=2, sort_keys=True, default=jsonify)


def calculate_statistics(folder: str, config: StatCalcConfig) -> JsonDict:
    """Calculates statistics of given folder and it's content

    :param folder: Path to folder for which statistics are being calculated
    :param config: A configuration of calculations
    """
    stats: JsonDict = {}

    for content in os.listdir(folder):
        content_path = fs.path.combine(folder, content)

        if os.path.isdir(content_path):
            fs_data_info = get_filesystem_data_info(OSFS("/"), content_path)
            if fs_data_info.bbox is not None:
                load_timestamps = fs_data_info.timestamps is not None
                eopatch = EOPatch.load(content_path, load_timestamps=load_timestamps)
                stats[content] = _calculate_eopatch_stats(eopatch, config)
            else:  # Probably it is not an EOPatch folder
                stats[content] = calculate_statistics(content_path, config)

        elif content_path.endswith("tiff"):
            stats[content] = _calculate_tiff_stats(content_path, config)
        elif content_path.endswith(".npy"):
            stats[content] = _calculate_numpy_stats(np.load(content_path, allow_pickle=True), config)
        elif content_path.endswith((".geojson", ".gpkg")):
            stats[content] = _calculate_vector_stats(gpd.read_file(content_path), config)
        elif content_path.endswith(".parquet"):
            stats[content] = _calculate_vector_stats(gpd.read_parquet(content_path), config)
        else:
            stats[content] = None

    return stats


def _calculate_eopatch_stats(eopatch: EOPatch, config: StatCalcConfig) -> JsonDict:
    """Calculates statistics of given EOPatch and it's content"""
    stats: JsonDict = defaultdict(dict)

    stats["bbox"] = repr(eopatch.bbox)
    if eopatch.timestamps is not None:
        stats["timestamps"] = [time.isoformat() for time in eopatch.timestamps]

    for ftype, fname in eopatch.get_features():
        if ftype.is_array():
            stats[ftype.value][fname] = _calculate_numpy_stats(eopatch[ftype, fname], config)
        elif ftype.is_vector():
            stats[ftype.value][fname] = _calculate_vector_stats(eopatch[ftype, fname], config)
        elif ftype is FeatureType.META_INFO:
            stats[ftype.value][fname] = str(eopatch[ftype, fname])

    return {**stats}


def _calculate_numpy_stats(raster: np.ndarray, config: StatCalcConfig) -> JsonDict:
    """Calculates statistics over a raster numpy array"""
    stats: JsonDict = {"array_shape": list(raster.shape), "dtype": str(raster.dtype)}
    if raster.dtype == object or raster.dtype.kind == "U":
        return stats

    unique_values = np.unique(raster)

    if unique_values.size <= config.unique_values_limit:
        values, counts = np.unique(raster, return_counts=True)
        stats["values"] = [
            {"value": _prepare_value(value, raster.dtype), "count": int(count)} for value, count in zip(values, counts)
        ]

    else:
        number_values = raster[~np.isnan(raster)]
        finite_values = number_values[np.isfinite(number_values)]

        stats["counts"] = {
            "nan": raster.size - number_values.size,
            "infinite": number_values.size - finite_values.size,
        }
        stats["basic_stats"] = _calculate_basic_stats(finite_values)

        # Randomly samples a small amount of points from the array (10% by default) to recalculate the statistics.
        # This introduces a 'positional instability' so that accidental mirroring or re-orderings are detected.
        rng = np.random.default_rng(0)
        subsample = rng.choice(finite_values, int(finite_values.size * 0.1))
        stats["subsample_basic_stats"] = _calculate_basic_stats(subsample)

        counts, edges = np.histogram(finite_values, bins=config.histogram_bin_num)
        stats["histogram"] = {
            "counts": counts.astype(int).tolist(),
            "edges": [_prepare_value(x, edges.dtype) for x in edges],  # type: ignore[arg-type]
        }

    if unique_values.size > 1:
        stats["random_values"] = _get_random_values(raster, config)

    return stats


def _calculate_tiff_stats(tiff_filename: str, config: StatCalcConfig) -> JsonDict:
    """Calculates statistics over a .tiff image"""
    with rasterio.open(tiff_filename) as tiff:
        return {
            "image": _calculate_numpy_stats(tiff.read(), config),
            "mask": _calculate_numpy_stats(tiff.dataset_mask(), config),
        }


def _calculate_vector_stats(gdf: gpd.GeoDataFrame, config: StatCalcConfig) -> JsonDict:
    """Calculates statistics over a vector GeoDataFrame"""

    def _rounder(coords: tuple[float, float] | Point) -> tuple[float, float]:
        x, y = (coords.x, coords.y) if isinstance(coords, Point) else coords
        return _prepare_value(x, np.float64), _prepare_value(y, np.float64)

    def _get_coords_sample(geom: Polygon | MultiPolygon | Any) -> list[tuple[float, float]] | None:
        if isinstance(geom, MultiPolygon):
            return _get_coords_sample(geom.geoms[0])
        return [_rounder(point) for point in geom.exterior.coords[:10]] if isinstance(geom, Polygon) else None

    stats = {
        "columns": list(gdf),
        "row_count": len(gdf),
        "crs": str(gdf.crs),
        "mean_area": _prepare_value(gdf.area.mean(), np.float64),
        "total_bounds": [_prepare_value(x, dtype=np.float64) for x in gdf.total_bounds],
    }

    if len(gdf):
        subsample: gpd.GeoDataFrame = gdf.sample(min(len(gdf), config.num_random_values), random_state=42)
        subsample["centroid"] = subsample.centroid.apply(_rounder)
        subsample["area"] = subsample.area.apply(_prepare_value, dtype=np.float64)
        subsample["geometry_type"] = subsample.geometry.geom_type
        subsample["some_coords"] = subsample.geometry.apply(_get_coords_sample)

        subsample_json_string = subsample.drop(columns="geometry").to_json(orient="index", date_format="iso")
        stats["random_rows"] = json.loads(subsample_json_string)

    return stats


def _calculate_basic_stats(values: np.ndarray) -> dict[str, float]:
    """Randomly samples a small amount of points from the array (10% by default) to recalculate the statistics.
    This introduces a 'positional instability' so that accidental mirroring or re-orderings are detected."""

    return {
        "min": _prepare_value(np.min(values), values.dtype),
        "max": _prepare_value(np.max(values), values.dtype),
        "mean": _prepare_value(np.mean(values), np.float32),
        "median": _prepare_value(np.median(values), np.float32),
        "std": _prepare_value(np.std(values), np.float32),
    }


def _get_random_values(raster: np.ndarray, config: StatCalcConfig) -> list[float]:
    """It randomly samples a few values from the array and marks their locations."""
    rng = np.random.default_rng(0)
    values = raster[np.isfinite(raster)]
    return [_prepare_value(x, values.dtype) for x in rng.choice(values.ravel(), config.num_random_values)]


def _prepare_value(value: Any, dtype: type) -> Any:
    """Converts a value in a way that it can be compared and serialized into a JSON. It also rounds float values."""
    if not np.isscalar(value):
        return value
    if not np.isfinite(value):
        return repr(value)
    if np.issubdtype(dtype, np.integer):
        value = cast(int, value)
        return int(value)
    if np.issubdtype(dtype, bool):
        return bool(value)
    value = cast(float, value)
    return float(f"{value:.5e}" if np.issubdtype(dtype, np.float32) else f"{value:.10e}")


def check_pipeline_logs(pipeline: Pipeline) -> None:
    """A utility function which checks pipeline logs and makes sure there are no failed executions"""
    if not pipeline.logging_manager.config.save_logs:
        raise ValueError("Pipeline did not save logs, this test would be useless")

    logs_folder = pipeline.logging_manager.get_pipeline_logs_folder(pipeline.current_execution_name)

    for filename in ["failed.json", "finished.json", "pipeline-report.json", "pipeline.log"]:
        path = fs.path.combine(logs_folder, filename)
        assert pipeline.storage.filesystem.isfile(path), f"File {path} is missing"

    failed_filename = fs.path.combine(logs_folder, "failed.json")
    assert not load_names(pipeline.storage.filesystem, failed_filename), f"Some executions failed, check {logs_folder}"

    finished_filename = os.path.join(logs_folder, "finished.json")
    assert load_names(pipeline.storage.filesystem, finished_filename), "No executions finished"


def run_config(
    config_path: str,
    *,
    output_folder_key: str | None = None,
    reset_output_folder: bool = True,
    check_logs: bool = True,
) -> str | None:
    """Runs a pipeline (or multiple) and checks the logs that all the executions were successful. Returns the full path
    of the output folder (if there is one) so it can be inspected further. In case of chain configs, the output folder
    of the last config is returned.

    :param config_path: A path to the config file
    :param output_folder_key: Type of the folder containing results of the pipeline, inferred from config if None
    :param reset_output_folder: Delete the content of the results folder before running the pipeline
    :param check_logs: If pipeline logs should be checked after the run completes. If EOWorkflows were used, the
        function fails if there were unsuccessful executions.
    """
    collected_configs = collect_configs_from_path(config_path)
    crude_configs = collected_configs if isinstance(collected_configs, list) else [collected_configs]
    raw_configs = [interpret_config_from_dict(config) for config in crude_configs]

    for config in raw_configs:
        output_folder_key = output_folder_key or config.get("output_folder_key")

        pipeline = load_pipeline_class(config).from_raw_config(config)

        if reset_output_folder:
            if output_folder_key is None:
                raise ValueError("Pipeline does not have an `output_folder_key` parameter, it must be set by hand.")
            folder = pipeline.storage.get_folder(output_folder_key)
            pipeline.storage.filesystem.removetree(folder)

        pipeline.run()

        if check_logs:
            check_pipeline_logs(pipeline)

    return pipeline.storage.get_folder(output_folder_key, full_path=True) if output_folder_key else None


def compare_content(
    folder_path: str | None,
    stats_path: str,
    *,
    save_new_stats: bool = False,
) -> None:
    """Compares the results from a pipeline run with the saved statistics. Constructed to be coupled with `run_config`
    hence the `Optional` input.

    :param folder_path: A path to the folder with contents to be compared
    :param stats_path: A path to the file containing result statistics
    :param save_new_stats: Save new result stats and skip the comparison
    """
    if folder_path is None:
        raise ValueError("The given path is None. The pipeline likely has no `output_folder_key` parameter.")

    stats = calculate_statistics(folder_path, config=StatCalcConfig())

    if save_new_stats:
        save_statistics(stats, stats_path)

    stats_difference = compare_with_saved(stats, stats_path)
    if stats_difference:
        stats_difference_repr = stats_difference.to_json(indent=2, sort_keys=True)
        raise AssertionError(f"Expected and obtained stats differ:\n{stats_difference_repr}")


def generate_tiff_file(
    filesystem: FS,
    file_paths: Iterable[str],
    *,
    tiff_bbox: BBox,
    width: int,
    height: int,
    num_bands: int,
    dtype: type,
    seed: int = 42,
) -> None:
    """Generates tiff files containing random data."""
    transform = rasterio.transform.from_bounds(*tiff_bbox, width=width, height=height)

    generator = np.random.default_rng(seed)
    shape = (num_bands, height, width) if num_bands is not None else (height, width)

    for path in file_paths:
        with filesystem.openbin(path, "w") as file_handle:
            with rasterio.open(
                file_handle,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=num_bands,
                dtype=dtype,
                nodata=0,
                transform=transform,
                crs=tiff_bbox.crs.ogc_string(),
            ) as dst:
                data = 10000 * generator.random(shape)
                dst.write(data)
