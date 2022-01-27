import pytest
from geopandas import GeoDataFrame

from sentinelhub import BBox, Geometry, CRS
from eogrow.core.config import Config
from eogrow.core.area import CustomGridAreaManager

pytestmark = pytest.mark.fast


CONFIG = Config.from_dict({"grid_filename": "test_custom_grid.geojson"})


def test_custom_grid_area_manager(storage):
    manager = CustomGridAreaManager(CONFIG, storage)

    grid = manager.get_grid(add_bbox_column=True)
    assert isinstance(grid, list)
    assert len(grid) == 1
    gdf = grid[0]
    assert isinstance(gdf, GeoDataFrame)
    assert gdf.crs.to_epsg() == 32638
    assert len(gdf.index) == 2
    assert gdf.BBOX.values[0] == BBox((729480.0, 4390045.0, 732120.0, 4391255.0), CRS.UTM_38N)

    area_gdf = manager.get_area_dataframe(crs=CRS.UTM_38N)
    assert area_gdf.equals(gdf.drop(columns=["BBOX"]))

    geometry = manager.get_area_geometry(crs=CRS.UTM_38N)
    assert geometry == Geometry(
        "POLYGON ((729480 4391145, 729480 4391255, 729480 4392355, 732120 4392355, 732120 4391255, 732120 4391145, "
        "732120 4390045, 729480 4390045, 729480 4391145))",
        crs=CRS.UTM_38N,
    )
