import pytest
from geopandas import GeoDataFrame

from sentinelhub import CRS, Geometry

from eogrow.core.area import UtmZoneAreaManager
from eogrow.core.area.utm import create_utm_zone_grid

LARGE_AREA_CONFIG = {
    "geometry_filename": "test_large_area.geojson",
    "patch": {"size_x": 1000000, "size_y": 1000000, "buffer_x": 0, "buffer_y": 0},
}


AREA_CONFIG = {
    "geometry_filename": "test_area.geojson",
    "patch": {"size_x": 2400, "size_y": 1100, "buffer_x": 120, "buffer_y": 55},
}


@pytest.mark.parametrize(
    ("config", "expected_zone_num", "expected_bbox_num"),
    [
        (AREA_CONFIG, 1, 2),
        (LARGE_AREA_CONFIG, 71, 368),
    ],
)
def test_bbox_split(storage, config, expected_zone_num, expected_bbox_num):
    area_manager = UtmZoneAreaManager.from_raw_config(config, storage)

    grid = area_manager.get_grid()

    assert isinstance(grid, dict)
    assert len(grid) == expected_zone_num

    bbox_count = 0
    for crs, subgrid in grid.items():
        assert crs == CRS(subgrid.crs)
        assert isinstance(subgrid, GeoDataFrame)
        bbox_count += len(subgrid.index)

    assert bbox_count == expected_bbox_num


def test_cache_name(storage):
    area_manager = UtmZoneAreaManager.from_raw_config(AREA_CONFIG, storage)

    assert area_manager.get_grid_cache_filename() == "UtmZoneAreaManager_test_area_2400_1100_120.0_55.0_0.0_0.0.gpkg"


def test_create_utm_zone_grid():
    area_geometry = Geometry(
        "POLYGON ((17.87338 47.88549, 17.87338 48.06064, 18.14598 48.06064, 18.14598 47.88549, 17.87338 47.88549))",
        crs=CRS.WGS84,
    )  # Covers UTM zones 33N and 34N

    name_column = "eopatch_name"
    bbox_size = (5000, 5000)
    bbox_offset = (0, 0)
    bbox_buffer = (0, 0)

    grid = create_utm_zone_grid(area_geometry, name_column, bbox_size, bbox_offset, bbox_buffer)

    assert isinstance(grid, dict)
    assert len(grid) == 2
    assert CRS("32633") in grid
    assert CRS("32634") in grid

    for gdf, n_expected_tiles in zip(grid.values(), (15, 15)):
        assert isinstance(gdf, GeoDataFrame)
        assert set(gdf.columns) == {"id", name_column, "geometry"}
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf.index) == n_expected_tiles
