"""
Tests for enum utils
"""
import pytest

from eogrow.utils.enum import BaseEOGrowEnum

pytestmark = pytest.mark.fast


class MyEnumClass(BaseEOGrowEnum):
    """Test Enum class containing basic crop types"""

    NO_DATA = "no data", 0, "#ffffff"
    SOME_DATA = "some data", 1, "#880000"
    ALL_DATA = "all data", 2, "#ff0000"


def test_evalscript_creation():
    evalscript = MyEnumClass.get_sentinel_hub_evaluation_function("band")
    assert evalscript is not None
    assert len(evalscript) > 0


def test_legend_creation():
    legend = MyEnumClass.get_sentinel_hub_legend()
    assert legend is not None
    assert len(legend) > 0


def test_properties():
    assert MyEnumClass.ALL_DATA.id == 2
    assert MyEnumClass.ALL_DATA.rgb_int[1] == 0
    assert MyEnumClass.NO_DATA.rgb_int[1] == 255
    assert MyEnumClass.NO_DATA.rgb_int[1] == 255
    assert MyEnumClass.SOME_DATA.nice_name == "Some data"
    assert MyEnumClass.SOME_DATA.color == "#880000"

    assert MyEnumClass.has_value("some data")
    assert MyEnumClass.has_value(0)
    assert MyEnumClass.has_value("#ff0000")
    assert not MyEnumClass.has_value("anything else")
