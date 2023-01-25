from typing import Any, Dict, Tuple

import pytest

from sentinelhub import CRS, BBox

from eogrow.tasks.spatial import get_array_slices

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "input_params, expected",
    [
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (731000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (730990, 4400650)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 99, None), slice(35, 100, None)),
            id="contained",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 200, None), slice(0, 100, None)),
            id="fully covered",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
                "limit_x": (50, 100),
                "limit_y": (100, 300),
            },
            (slice(50, 100, None), slice(100, 100, None)),
            id="test limits",
        ),
        pytest.param(
            {
                "bbox": BBox(bbox=[500, 500, 600, 600], crs=CRS.WGS84),
                "slice_bbox": BBox(bbox=[520, 520, 580, 590], crs=CRS.WGS84),
                "resolution": (10, 10),
            },
            (slice(2, 8, None), slice(1, 8, None)),
            id="test WGS84",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((731000, 4400000), (732000, 4400500)), crs=CRS("32638")),
                "size": (10, 10),
            },
            (slice(5, 10, None), slice(5, 10, None)),
            id="test size",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "size": (100, 100),
                "limit_x": (50, 100),
                "limit_y": (100, 200),
            },
            (slice(50, 100, None), slice(100, 100, None)),
            id="test size limit",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730100, 4400100), (731900, 4400800)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(10, 190, None), slice(20, 90, None)),
            id="test slice_bbox included",
        ),
        pytest.param(
            # Test is specific to current implementation, serves only to detect unplanned changes.
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((731000, 4400500), (733000, 4401500)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(100, 300, None), slice(-50, 50, None)),
            id="test slice_bbox intersection",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "size": (10, 10),
                "limit_x": (50, 100),
                "limit_y": (100, 200),
            },
            (slice(50, 10, None), slice(100, 10, None)),
            id="unintended behaviour",
            marks=pytest.mark.skip("unintended behaviour"),
        ),
    ],
)
def test_get_array_slices(input_params: Dict[str, Any], expected: Tuple[slice, slice]) -> None:
    assert get_array_slices(**input_params) == expected


BBOX_FOR_INVALID_INPUT: BBox = BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638"))


@pytest.mark.parametrize(
    "input_params",
    [
        pytest.param(
            {
                "bbox": BBOX_FOR_INVALID_INPUT,
                "slice_bbox": BBOX_FOR_INVALID_INPUT,
            },
            id="no resolution or size",
        ),
        pytest.param(
            {
                "bbox": BBOX_FOR_INVALID_INPUT,
                "slice_bbox": BBOX_FOR_INVALID_INPUT,
                "resolution": (10, 10),
                "size": (10, 10),
            },
            id="resolution and size",
        ),
        pytest.param(
            {
                "bbox": BBOX_FOR_INVALID_INPUT,
                "slice_bbox": BBox(((730000.5, 4400000.5), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            id="clipping value not integer",
        ),
    ],
)
def test_get_array_slices_invalid_input(input_params: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        get_array_slices(**input_params)
