import random

import pytest
from geopandas import GeoDataFrame
from shapely.ops import unary_union

from sentinelhub import CRS, BBox

from eogrow.utils.grid import GridTransformation, create_transformations, get_enclosing_bbox, get_grid_bbox, split_bbox

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


@pytest.mark.parametrize(
    "bbox, split_x, split_y, buffer_x, buffer_y",
    (
        [BBox([10, 10, 20, 20], CRS.WGS84), 2, 3, 0, 0],
        [BBox([1000, 500, 2000, 800], CRS(32612)), 5, 2, 50, 100],
        [BBox([0.3, 5.2, 12.234, 7.2315], CRS.WGS84), 1, 3, 0.2, 0],
    ),
)
def test_split_bbox_basics(bbox: BBox, split_x: int, split_y: int, buffer_x: float, buffer_y: float) -> None:
    """Checks that the split produces correct amount of bboxes and that the geometries cover the original."""
    bbox_split = split_bbox(("test", bbox), split_x, split_y, buffer_x, buffer_y)
    assert len(bbox_split) == split_x * split_y

    merged_bboxes = unary_union([bbox.geometry for _, bbox in bbox_split])
    assert bbox.geometry.equals(merged_bboxes)


def test_split_bbox_buffer_and_names() -> None:
    """Checks that buffers are applied correctly and that the naming schema works as intended."""
    bbox = BBox([9, 9, 17, 17], CRS.WGS84)
    split = {name: bbox for name, bbox in split_bbox(("test", bbox), 3, 3, 1, 1, "{name}{i_x}{i_y}")}
    expected = {
        "test00": BBox([9, 9, 13, 13], CRS.WGS84),
        "test10": BBox([11, 9, 15, 13], CRS.WGS84),
        "test20": BBox([13, 9, 17, 13], CRS.WGS84),
        "test01": BBox([9, 11, 13, 15], CRS.WGS84),
        "test11": BBox([11, 11, 15, 15], CRS.WGS84),
        "test21": BBox([13, 11, 17, 15], CRS.WGS84),
        "test02": BBox([9, 13, 13, 17], CRS.WGS84),
        "test12": BBox([11, 13, 15, 17], CRS.WGS84),
        "test22": BBox([13, 13, 17, 17], CRS.WGS84),
    }
    assert split == expected
