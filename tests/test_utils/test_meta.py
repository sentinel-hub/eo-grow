import datetime as dt
import os

import pytest

from eogrow import __version__
from eogrow.core.pipeline import Pipeline
from eogrow.core.schemas import BaseSchema
from eogrow.core.storage import StorageManager
from eogrow.utils.meta import (
    collect_schema,
    get_os_import_path,
    get_package_versions,
    import_object,
    load_pipeline_class,
)

pytestmark = pytest.mark.fast


def test_load_pipeline_class():
    pipeline_class = load_pipeline_class({"pipeline": "eogrow.core.pipeline.Pipeline"})
    assert pipeline_class is Pipeline


def test_load_pipeline_class_missing_param():
    with pytest.raises(ValueError):
        load_pipeline_class({"x": "y"})


def test_load_pipeline_class_wrong_import_path():
    with pytest.raises(ValueError):
        load_pipeline_class({"pipeline": "wrong-import-path!"})


@pytest.mark.parametrize("object_with_schema", [Pipeline, StorageManager])
def test_collect_schema(object_with_schema):
    schema = collect_schema(object_with_schema)
    assert issubclass(schema, BaseSchema)


def test_import_object():
    dt_class = import_object("datetime.datetime")
    assert dt_class is dt.datetime


def test_import_object_from_wrong_module():
    with pytest.raises(ModuleNotFoundError):
        import_object("xyz.datetime")


def test_import_object_with_wrong_name():
    with pytest.raises(ImportError):
        import_object("datetime.xyz")


def test_get_os_import_path_from_3rd_party():
    path = get_os_import_path("eogrow")
    assert path.endswith("__init__.py")
    assert os.path.dirname(path).endswith("eogrow")


def test_get_os_import_path_from_std_lib():
    path = get_os_import_path("datetime")
    assert path.endswith("datetime.py")


def test_get_os_import_path_from_non_existing():
    with pytest.raises(ValueError):
        get_os_import_path("package-that-doesnt-exist!")


def test_get_package_versions():
    versions = get_package_versions()

    assert isinstance(versions, dict)
    assert "error" not in versions
    assert versions["eo-grow"] == __version__
    assert len(versions) > 10
