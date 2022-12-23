from sentinelhub import CRS, BBox

from eogrow.utils.eopatch_list import group_by_crs


def test_group_by_crs() -> None:
    names = ["beep", "boop", "bap", "skibidi", "bapap"]
    bboxes = [
        BBox((0, 0, 1, 1), CRS.WGS84),
        BBox((0, 0, 1, 1), CRS.POP_WEB),
        BBox((0, 0, 1, 1), CRS.WGS84),
        BBox((0, 0, 1, 1), CRS(35035)),
        BBox((0, 0, 1, 1), CRS(35035)),
    ]
    patch_list = list(zip(names, bboxes))
    grouped_patches = group_by_crs(patch_list)
    expected = {CRS.WGS84: ["beep", "bap"], CRS.POP_WEB: ["boop"], CRS(35035): ["skibidi", "bapap"]}

    assert len(expected) == len(grouped_patches)

    for crs, expected_values in expected.items():
        assert sorted(expected_values) == sorted(grouped_patches[crs])
