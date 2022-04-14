import os
from tempfile import NamedTemporaryFile

import pytest

from eogrow.core.area import CustomGridAreaManager, UtmZoneAreaManager
from eogrow.core.config import interpret_config_from_dict, interpret_config_from_path
from eogrow.core.eopatch import CustomGridEOPatchManager, EOPatchManager

pytestmark = pytest.mark.fast


@pytest.fixture(name="eopatch_manager")
def eopatch_manager_fixture(storage, config):
    area_manager = UtmZoneAreaManager.from_raw_config(config["area"], storage)
    return EOPatchManager.from_raw_config(config["eopatch"], area_manager)


@pytest.fixture(name="special_eopatch_manager")
def special_eopatch_manager_fixture(storage, config_folder):
    path = os.path.join(config_folder, "other", "eopatch_global_config.json")
    config = interpret_config_from_path(path)
    area_manager = UtmZoneAreaManager.from_raw_config(config["area"], storage)
    return EOPatchManager.from_raw_config(config["eopatch"], area_manager)


@pytest.fixture(name="filtered_eopatch_manager")
def filtered_eopatch_manager_fixture(storage, config_folder):
    path = os.path.join(config_folder, "other", "eopatch_global_config_filtered.json")
    config = interpret_config_from_path(path)
    area_manager = UtmZoneAreaManager.from_raw_config(config["area"], storage)
    return EOPatchManager.from_raw_config(config["eopatch"], area_manager)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_eopatch_filenames(eopatch_manager):

    assert eopatch_manager.get_eopatch_filenames()[0] == "eopatch-id-0-col-0-row-0"

    for folder, id_list, filter_existing, list_size in [
        (None, None, False, 2),
        ("something", [1, 0, 0, 1], False, 4),
        (".", None, True, 0),
    ]:

        eopatch_list = eopatch_manager.get_eopatch_filenames(
            folder=folder, id_list=id_list, filter_existing=filter_existing
        )
        assert isinstance(eopatch_list, list)
        assert len(eopatch_list) == list_size
        for name in eopatch_list:
            assert isinstance(name, str)
            assert os.path.basename(name).startswith("eopatch-id")
            assert eopatch_manager.is_eopatch_name(name)


def test_eopatch_naming(special_eopatch_manager):
    eopatch_names = special_eopatch_manager.get_eopatch_filenames()

    assert len(eopatch_names) == 536

    assert eopatch_names[0] == "eopatch-id-000-col-0-row-11"
    assert eopatch_names[45] == "eopatch-id-045-col-3-row-14"
    assert eopatch_names[-1] == "eopatch-id-615-col-26-row-18"

    assert special_eopatch_manager.get_eopatch_filenames(id_list=[615])[0] == "eopatch-id-615-col-26-row-18"

    with pytest.raises(KeyError):
        special_eopatch_manager.get_eopatch_filenames(id_list=[600])


def test_eopatch_filtered_naming(filtered_eopatch_manager):
    eopatch_names = filtered_eopatch_manager.get_eopatch_filenames()
    assert eopatch_names == ["eopatch-id-000-col-0-row-11", "eopatch-id-001-col-0-row-12"]

    filtered_names = filtered_eopatch_manager.get_eopatch_filenames(id_list=[1])
    assert filtered_names == ["eopatch-id-001-col-0-row-12"]

    with pytest.raises(KeyError):
        filtered_eopatch_manager.get_eopatch_filenames(id_list=[600])


def test_saving_loading(eopatch_manager):
    all_names = eopatch_manager.get_eopatch_filenames()

    for file_format in [".json", ".txt"]:
        for eopatch_list, id_list, expected_names in [
            (None, None, all_names),
            (all_names[:-1], None, all_names[:-1]),
            (all_names[:2], [0], all_names[:1]),
            ([1, 0, 1], None, [all_names[1], all_names[0], all_names[1]]),
            ([0, 0], [1], []),
        ]:
            with NamedTemporaryFile(suffix=file_format) as temp:
                eopatch_manager.save_eopatch_filenames(temp.name, eopatch_list=eopatch_list)

                assert os.stat(temp.name).st_size > 0

                loaded_names = eopatch_manager.load_eopatch_filenames(temp.name, id_list=id_list)

                assert isinstance(loaded_names, list)
                assert len(loaded_names) == len(expected_names)
                for name1, name2 in zip(loaded_names, expected_names):
                    assert name1 == name2

    with pytest.raises(ValueError):
        eopatch_manager.save_eopatch_filenames("wrong-format-file.shp")
    with pytest.raises(ValueError):
        eopatch_manager.load_eopatch_filenames("wrong-format-file.tiff")


CUSTOM_GRID_AREA_CONFIG = interpret_config_from_dict({"grid_filename": "test_custom_grid.geojson"})
CUSTOM_GRID_EOPATCH_CONFIG = interpret_config_from_dict({"name_column": "name", "index_column": "index_n"})


def test_custom_grid_eopatch_manager(storage):
    area_manager = CustomGridAreaManager.from_raw_config(CUSTOM_GRID_AREA_CONFIG, storage)
    eopatch_manager = CustomGridEOPatchManager.from_raw_config(CUSTOM_GRID_EOPATCH_CONFIG, area_manager)

    eopatch_names = eopatch_manager.get_eopatch_filenames()
    assert eopatch_names == ["patch0", "patch1"]
