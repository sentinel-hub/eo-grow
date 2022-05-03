"""
Utilities for exporting data
"""
import logging
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
from fs.base import FS

from sentinelhub import CRS, BBox

from .fs import LocalFile

LOGGER = logging.getLogger(__name__)


def export_grid_stats(
    stats_list: List[Dict[str, object]],
    bbox_list: List[BBox],
    path: str,
    filesystem: Optional[FS] = None,
    names: Optional[List[str]] = None,
) -> None:
    """Exports stats per each bounding box (i.e. EOPatch) into a Geopackage file

    :param stats_list: Dictionaries of statistical values and names, one dictionary per each bounding box
    :param bbox_list: Bounding boxes that correspond to stats in `stats_list`
    :param path: A path to a Geopackage file into which data will be exported. In case filesystem object is provided it
        should be a relative path otherwise an absolute path
    :param filesystem: A filesystem object.
    :param names: If provided these names will be included as an additional column in the exported Geopackage
    """
    if len(stats_list) != len(bbox_list):
        raise ValueError(
            f"stats_list and bbox_list should have the same length but found {len(stats_list)} and {len(bbox_list)}"
        )
    if names and len(stats_list) != len(names):
        raise ValueError(
            f"stats_list and names should have the same length but found {len(stats_list)} and {len(names)}"
        )

    data_per_crs: Dict[CRS, List[Dict[str, Any]]] = {}
    for index, (stats, bbox) in enumerate(zip(stats_list, bbox_list)):
        data_dict = {"geometry": bbox.geometry, **stats}
        if names:
            data_dict = {"name": names[index], **data_dict}

        data_per_crs[bbox.crs] = data_per_crs.get(bbox.crs, [])
        data_per_crs[bbox.crs].append(data_dict)

    grid = []
    for crs, data in data_per_crs.items():
        gdf = gpd.GeoDataFrame(data=data, crs=crs.pyproj_crs())

        # Cast objects to strings, otherwise they cannot be serialized into a Geopackage
        for column_name in gdf.columns:
            if gdf[column_name].dtype == np.dtype("object"):
                gdf[column_name] = gdf[column_name].astype(str)

        grid.append(gdf)

    with LocalFile(path, mode="w", filesystem=filesystem) as local_file:
        for gdf in grid:
            gdf.to_file(
                local_file.path,
                driver="GPKG",
                encoding="utf-8",
                layer=f"Stats for EPSG:{gdf.crs.to_epsg()}",
            )
    LOGGER.info("Saved stats to %s", path)
