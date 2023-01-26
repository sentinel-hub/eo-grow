import pytest
from shapely.ops import unary_union

from sentinelhub import CRS, BBox

from eogrow.utils.grid import split_bbox


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
