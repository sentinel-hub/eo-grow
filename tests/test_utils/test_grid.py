"""
Tests of grid utilities
"""
import random

import pytest
from geopandas import GeoDataFrame

from sentinelhub import CRS, BBox

from eogrow.utils.grid import GridTransformation, create_transformations, get_enclosing_bbox, get_grid_bbox

pytestmark = pytest.mark.fast

DEFAULT_BBOX = BBox((0, 0, 1, 1), CRS.WGS84)


def test_grid_transformation():
    bbox = BBox((0, 0, 1, 1), CRS.WGS84)

    transformation = GridTransformation(
        enclosing_bbox=bbox,
        source_bboxes=(bbox,),
        target_bboxes=(bbox, bbox),
    )
    assert isinstance(transformation, GridTransformation)

    with pytest.raises(ValueError):
        GridTransformation(
            enclosing_bbox=bbox,
            source_bboxes=(bbox,),
            target_bboxes=(),
        )


@pytest.mark.parametrize(
    "source_size, target_size",
    [
        (10, 18),
        (18, 10),
        (100, 100),
    ],
)
def test_create_transformations(source_size, target_size):
    source_gdf = _create_gdf(source_size)
    target_gdf = _create_gdf(target_size)

    min_size = min(source_size, target_size)
    source_gdf.index_n = source_gdf.index_n % min_size
    target_gdf.index_n = target_gdf.index_n % min_size
    target_gdf.index_n.values[-1] = min_size  # Creates 1 group with no source bboxes

    transformations = create_transformations([source_gdf], [target_gdf], match_columns=["index_n"])

    assert isinstance(transformations, list)
    assert all(isinstance(transformation, GridTransformation) for transformation in transformations)

    expected_count = min(source_size + 1, target_size)
    assert len(transformations) == expected_count

    assert transformations[-1].source_bboxes == ()


def _create_gdf(size: int) -> GeoDataFrame:
    """Creates a dummy GeoDataFrame of the given size"""
    info_list = [{"index_n": index, "col1": 10, "col2": "x", "BBOX": DEFAULT_BBOX} for index in range(size)]
    return GeoDataFrame(
        info_list, geometry=[DEFAULT_BBOX.geometry for _ in range(size)], crs=DEFAULT_BBOX.crs.pyproj_crs()
    )


def test_get_grid_box():
    bbox = BBox((1, 2, 2, 3), CRS.WGS84)
    grid_bbox = get_grid_bbox(bbox, index=(1, 2), split=(10, 10))

    assert grid_bbox == BBox((0, 0, 10, 10), CRS.WGS84)


INDICES = list(range(10))
random.shuffle(INDICES)
RANDOM_BBOXES = [BBox((i, j, i + 1, j + 1), CRS.WGS84) for i, j in zip(range(10), INDICES)]


@pytest.mark.parametrize(
    "bboxes, expected_enclosing_bbox",
    [
        ([BBox((0, 0, 1, 1), CRS.WGS84), BBox((1, 1, 2, 2), CRS.WGS84)], BBox((0, 0, 2, 2), CRS.WGS84)),
        ([BBox((0, 0, 2, 1), CRS.WGS84), BBox((0, 0, 1, 2), CRS.WGS84)], BBox((0, 0, 2, 2), CRS.WGS84)),
        (RANDOM_BBOXES, BBox((0, 0, 10, 10), CRS.WGS84)),
    ],
)
def test_get_enclosing_bbox(bboxes, expected_enclosing_bbox):
    enclosing_bbox = get_enclosing_bbox(bboxes)
    assert enclosing_bbox == expected_enclosing_bbox


def test_get_enclosing_bbox_errors():
    with pytest.raises(ValueError):
        get_enclosing_bbox([])

    bboxes_mixed_crs = [
        BBox((0, 0, 1, 1), CRS.WGS84),
        BBox((0, 0, 1, 1), CRS.POP_WEB),
    ]
    with pytest.raises(ValueError):
        get_enclosing_bbox(bboxes_mixed_crs)
