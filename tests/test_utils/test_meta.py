import datetime as dt

import pytest

from eogrow.utils.meta import import_object

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
