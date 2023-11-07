"""
Module containing utilities for working with vector data
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Polygon

from sentinelhub import CRS


def concat_gdf(dataframe_list: list[GeoDataFrame], reproject_crs: CRS | None = None) -> GeoDataFrame:
    """Concatenates together multiple GeoDataFrames, all in the same CRS

    There exists pandas.concat but no geopandas.concat. Therefore, this function implements it.

    :param dataframe_list: A list of GeoDataFrames to be concatenated together
    :param reproject_crs: A CRS in which dataframes should be reprojected before being joined
    :return: A joined GeoDataFrame
    """
    if not (dataframe_list and all(isinstance(gdf, GeoDataFrame) for gdf in dataframe_list)):
        raise ValueError(f"Expected a list of GeoDataFrames with at least 1 element, got {dataframe_list}")

    if reproject_crs is not None:
        crs = reproject_crs.pyproj_crs()
        dataframe_list = [gdf.to_crs(crs) for gdf in dataframe_list]
    else:
        unique_crs = {vector_gdf.crs for vector_gdf in dataframe_list}
        if len(unique_crs) > 1:
            raise ValueError("GeoDataFrames are in different CRS, therefore `reproject_crs` parameter should be given")

    return gpd.GeoDataFrame(pd.concat(dataframe_list, ignore_index=True), crs=dataframe_list[0].crs)


def count_points(geometry: Polygon | MultiPolygon) -> int:
    """Counts a number of points for a given geometry, both from exterior and interiors"""
    if isinstance(geometry, MultiPolygon):
        return sum(count_points(subgeometry) for subgeometry in geometry.geoms)
    exterior_count = len(geometry.exterior.coords)
    interior_count = sum(len(interior.coords) for interior in geometry.interiors)
    return exterior_count + interior_count
