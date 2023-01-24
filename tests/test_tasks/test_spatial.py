from typing import Any, Dict, Tuple

import pytest

from sentinelhub import CRS, BBox

from eogrow.tasks.spatial import get_array_slices


@pytest.mark.parametrize(
    "input_params, expected",
    [
        (
            {
                "bbox": BBox(((730000, 4400000), (731000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (731000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 100, None), slice(0, 100, None)),
        ),
        (
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 200, None), slice(0, 100, None)),
        ),
        (
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
                "limit_x": (50, 100),
                "limit_y": (100, 300),
            },
            (slice(50, 100, None), slice(100, 100, None)),
        ),
        (
            {
                "bbox": BBox(bbox=[500, 500, 600, 600], crs=CRS.WGS84),
                "slice_bbox": BBox(bbox=[500, 500, 600, 600], crs=CRS.WGS84),
                "resolution": (10, 10),
            },
            (slice(0, 10, None), slice(0, 10, None)),
        ),
        (
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "size": (10, 10),
            },
            (slice(0, 10, None), slice(0, 10, None)),
        ),
        (
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "size": (10, 10),
                "limit_x": (50, 100),
                "limit_y": (100, 200),
            },
            (slice(50, 10, None), slice(100, 10, None)),
        ),
    ],
    ids=["clean_test", "random_test", "test_limits", "test_WGS84", "test_size", "test_size_limit"],
)
def test_get_array_slices(input_params: Dict[str, Any], expected: Tuple[slice, slice]) -> None:
    assert get_array_slices(**input_params) == expected


@pytest.mark.parametrize(
    "input_params",
    [
        {
            "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
            "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
        },
        {
            "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
            "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
            "resolution": (10, 10),
            "size": (10, 10),
        },
        {
            "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
            "slice_bbox": BBox(((730000.5, 4400000.5), (732000, 4401000)), crs=CRS("32638")),
            "resolution": (10, 10),
        },
    ],
    ids=["no_resolution_or_size", "resolution_and_size", "clipping_value_not_integer"],
)
def test_get_array_slices_invalid_input(input_params: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        get_array_slices(**input_params)
