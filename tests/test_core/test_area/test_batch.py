import os

import pytest
from geopandas import GeoDataFrame

from sentinelhub import Geometry

from eogrow.core.area import BatchAreaManager
from eogrow.core.area.batch import MissingBatchIdError
from eogrow.core.config import interpret_config_from_path

pytestmark = pytest.mark.fast


@pytest.fixture(scope="function", name="batch_config")
def large_area_config_fixture(config_folder):
    filename = os.path.join(config_folder, "other", "batch_area_config.json")
    return interpret_config_from_path(filename)


def test_area_shape(storage, batch_config):
    manager = BatchAreaManager.from_raw_config(batch_config, storage)

    area_dataframe = manager.get_area_dataframe()

    assert isinstance(area_dataframe, GeoDataFrame)
    assert len(area_dataframe.index) == 6

    geometry = manager.get_area_geometry()
    assert isinstance(geometry, Geometry)


def test_basics(storage, batch_config):
    manager = BatchAreaManager.from_raw_config(batch_config, storage)

    assert manager.batch_id == ""
    assert manager.subsplit == (1, 1)
    assert manager.absolute_buffer == (1200.0, 120.0)

    with pytest.raises(MissingBatchIdError):
        manager.get_grid()
