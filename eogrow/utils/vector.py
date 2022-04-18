"""
Module containing useful utilities for working with vector data
"""
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Polygon

from sentinelhub import CRS


def concat_gdf(dataframe_list: List[GeoDataFrame], reproject_crs: Union[CRS, int, None] = None) -> GeoDataFrame:
    """Concatenates together multiple GeoDataFrames, all in the same CRS

    There exists pandas.concat but no geopandas.concat. Therefore, this function implements it.

    :param dataframe_list: A list of GeoDataFrames to be concatenated together
    :param reproject_crs: A CRS in which dataframes should be reprojected before being joined
    :return: A joined GeoDataFrame
    """
    if not (dataframe_list and all(isinstance(gdf, GeoDataFrame) for gdf in dataframe_list)):
        raise ValueError(f"Expected a list of GeoDataFrames with at least 1 element, got {dataframe_list}")

    if reproject_crs:
        crs = CRS(reproject_crs).pyproj_crs()
        dataframe_list = [gdf.to_crs(crs) for gdf in dataframe_list]
    else:
        unique_crs = set(vector_gdf.crs for vector_gdf in dataframe_list)
        if len(unique_crs) > 1:
            raise ValueError("GeoDataFrames are in different CRS, therefore reproject_crs parameter should be given")

    return gpd.GeoDataFrame(pd.concat(dataframe_list, ignore_index=True), crs=dataframe_list[0].crs)


def filtered_sjoin(gdf1: GeoDataFrame, gdf2: GeoDataFrame) -> GeoDataFrame:
    """Performs spatial joins of 2 GeoDataFrames and filters repetitions of geometries in the resulting GeoDataFrame.
    Note that geometries in the resulting GeoDataFrame are from the first input GeoDataFrame.

    :param gdf1: First input data frame
    :param gdf2: Second input data frame
    :return: Resulting data frame
    """
    joined_gdf = gpd.sjoin(gdf1, gdf2)
    joined_gdf = joined_gdf.drop(columns=["index_right"])

    unique_index_name = _get_new_unique_column_name(joined_gdf)
    joined_gdf[unique_index_name] = joined_gdf.index.values

    filtered_joined_gdf = joined_gdf.groupby(unique_index_name).first()
    filtered_joined_gdf.index.name = None

    return filtered_joined_gdf


def spatial_difference(gdf1: GeoDataFrame, gdf2: GeoDataFrame) -> GeoDataFrame:
    """Removes polygons from the first GeoDataFrame that intersect with polygons from the second GeoDataFrame

    :param gdf1: First input data frame
    :param gdf2: Second input data frame
    :return: Resulting data frame
    """
    gdf2 = gdf2[["geometry"]]

    intersections = gpd.sjoin(gdf1, gdf2, how="left")
    result_gdf = intersections[intersections["index_right"].isna()]

    result_gdf = result_gdf.drop(columns=["index_right"])
    return result_gdf


def filter_intersecting(gdf: GeoDataFrame, filter_column: Optional[str] = None) -> GeoDataFrame:
    """Filters out intersecting geometries. A geometry is filtered out if it intersects a geometry with a lower index.

    E.g.: geometries 1 and 2 intersect and geometries 2 and 3 intersect. Then both geometries 2 and 3 will be removed.

    :param gdf: A dataframe to be filtered
    :param filter_column: A name of a column containing indices of geometries, according which to filter. If not
        provided it will use the dataframe index column
    """
    if filter_column is None:
        gdf = gdf.copy(deep=False)
        filter_column = _get_new_unique_column_name(gdf)
        gdf[filter_column] = gdf.index.values

    columns_right = ["geometry", filter_column]
    intersection_gdf = gpd.sjoin(gdf, gdf[columns_right])

    filter_left_column, filter_right_column = f"{filter_column}_left", f"{filter_column}_right"
    valid_column = _get_new_unique_column_name(intersection_gdf)
    intersection_gdf[valid_column] = intersection_gdf[filter_left_column] <= intersection_gdf[filter_right_column]

    intersection_gdf = intersection_gdf.drop(columns=["index_right", filter_left_column, filter_right_column])

    index_column = _get_new_unique_column_name(intersection_gdf)
    intersection_gdf[index_column] = intersection_gdf.index.values

    valid_only_gdf = intersection_gdf.groupby(index_column).filter(lambda df: df[valid_column].all())

    valid_only_gdf = valid_only_gdf.drop(columns=[valid_column])

    filtered_gdf = valid_only_gdf.groupby(index_column).first()
    filtered_gdf.index.name = None

    return filtered_gdf


def _get_new_unique_column_name(df: pd.DataFrame, prefix: str = "temp") -> str:
    """Provides a new column name that doesn't yet exist in a given data frame. The new column will be named in
    a form of <prefix><index>.
    """
    prefix_len = len(prefix)
    suffixes = {column[prefix_len:] for column in df.columns if column.startswith(prefix)}

    index = 1
    while str(index) in suffixes:
        index += 1

    return f"{prefix}{index}"


def count_points(geometry: Union[Polygon, MultiPolygon]) -> int:
    """Counts a number of points for a given geometry, both from exterior and interiors"""
    if isinstance(geometry, Polygon):
        exterior_count = len(geometry.exterior.coords)
        interior_count = sum(len(interior.coords) for interior in geometry.interiors)
        return exterior_count + interior_count

    if isinstance(geometry, MultiPolygon):
        return sum(count_points(subgeometry) for subgeometry in geometry.geoms)

    raise NotImplementedError(f"Counting points for geometry type {type(geometry)} is not yet supported")
