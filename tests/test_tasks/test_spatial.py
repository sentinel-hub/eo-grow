from typing import Any, Dict, Tuple

import pytest

from sentinelhub import CRS, BBox

from eogrow.tasks.spatial import get_array_slices


@pytest.mark.parametrize(
    "input_params, expected",
    [
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (731000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (731000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 100, None), slice(0, 100, None)),
            id="fully covered",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (731000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (730990, 4400650)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 99, None), slice(0, 65, None)),
            id="contained",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
            },
            (slice(0, 200, None), slice(0, 100, None)),
            id="random test",
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
                "slice_bbox": BBox(bbox=[500, 500, 600, 600], crs=CRS.WGS84),
                "resolution": (10, 10),
            },
            (slice(0, 10, None), slice(0, 10, None)),
            id="test WGS84",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "size": (10, 10),
            },
            (slice(0, 10, None), slice(0, 10, None)),
            id="test size",
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
            id="test size limit",
        ),
    ],
)
def test_get_array_slices(input_params: Dict[str, Any], expected: Tuple[slice, slice]) -> None:
    assert get_array_slices(**input_params) == expected


@pytest.mark.parametrize(
    "input_params",
    [
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
            },
            id="no resolution or size",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "slice_bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
                "resolution": (10, 10),
                "size": (10, 10),
            },
            id="resolution and size",
        ),
        pytest.param(
            {
                "bbox": BBox(((730000, 4400000), (732000, 4401000)), crs=CRS("32638")),
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
