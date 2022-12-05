import pytest
from geopandas import GeoDataFrame

from sentinelhub import CRS, BBox, Geometry

from eogrow.core.area import CustomGridAreaManager, NewCustomGridAreaManager
from eogrow.core.config import interpret_config_from_dict

pytestmark = pytest.mark.fast


CONFIG = interpret_config_from_dict({"grid_filename": "test_custom_grid.geojson"})


def test_custom_grid_area_manager(storage):
    manager = CustomGridAreaManager.from_raw_config(CONFIG, storage)

    grid = manager.get_grid(add_bbox_column=True)
    assert isinstance(grid, list)
    assert len(grid) == 1
    gdf = grid[0]
    assert isinstance(gdf, GeoDataFrame)
    assert gdf.crs.to_epsg() == 32638
    assert len(gdf.index) == 2
    assert gdf.BBOX.values[0] == BBox((729480.0, 4390045.0, 732120.0, 4391255.0), CRS.UTM_38N)

    geometry = manager.get_area_geometry(crs=CRS.UTM_38N)
    assert geometry == Geometry(
        "POLYGON ((729480 4391145, 729480 4391255, 729480 4392355, 732120 4392355, 732120 4391255, 732120 4391145, "
        "732120 4390045, 729480 4390045, 729480 4391145))",
        crs=CRS.UTM_38N,
    )


def test_new_custom_grid_area_manager(storage):
    config = interpret_config_from_dict({"grid_filename": "test_custom_grid.geojson", "name_column": "name"})
    manager = NewCustomGridAreaManager.from_raw_config(config, storage)

    grid = manager.get_grid()
    assert len(grid) == 1

    gdf = grid[CRS.UTM_38N]
    assert gdf.crs.to_epsg() == 32638
    assert len(gdf.index) == 2
    name_to_geom = {row.eopatch_name: row.geometry for _, row in gdf.iterrows()}
    assert name_to_geom["patch0"].bounds == (729480.0, 4390045.0, 732120.0, 4391255.0)

    geometry = manager.get_area_geometry(crs=CRS.UTM_38N)
    assert geometry == Geometry(
        "POLYGON ((729480 4391145, 729480 4391255, 729480 4392355, 732120 4392355, 732120 4391255, 732120 4391145, "
        "732120 4390045, 729480 4390045, 729480 4391145))",
        crs=CRS.UTM_38N,
    )

    assert manager.get_grid_cache_filename() == "NewCustomGridAreaManager_test_custom_grid.gpkg"
