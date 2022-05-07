import datetime as dt

import pytest

from eogrow import __version__
from eogrow.utils.meta import get_package_versions, import_object

pytestmark = pytest.mark.fast


def test_import_object():
    dt_class = import_object("datetime.datetime")
    assert dt_class is dt.datetime


def test_import_object_from_wrong_module():
    with pytest.raises(ModuleNotFoundError):
        import_object("xyz.datetime")


def test_import_object_with_wrong_name():
    with pytest.raises(ImportError):
        import_object("datetime.xyz")


def test_get_package_versions():
    versions = get_package_versions()

    assert isinstance(versions, dict)
    assert "error" not in versions
    assert versions["eo-grow"] == __version__
    assert len(versions) > 10
