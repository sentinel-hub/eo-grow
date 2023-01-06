import datetime as dt
from contextlib import nullcontext
from typing import Optional, Union

import numpy as np
import pytest
from pydantic import ValidationError

from sentinelhub import DataCollection
from sentinelhub.data_collections_bands import Band, MetaBands, Unit

from eogrow.core.schemas import BaseSchema, ManagerSchema
from eogrow.utils.types import RawSchemaDict
from eogrow.utils.validators import (
    ensure_defined_together,
    ensure_exactly_one_defined,
    field_validator,
    optional_field_validator,
    parse_data_collection,
    parse_dtype,
    parse_time_period,
    validate_manager,
)

pytestmark = pytest.mark.fast


def is_large_enough(value: int):
    """Checks size of integer"""
    assert value > 5
    return value


def parse_float(value, values) -> Optional[float]:
    """Extracts float from string and adds another field (to test `values` parameter as well)"""
    if isinstance(value, str):
        return float(value) + values.get("int_field", 0)
    return None


class DummySchema(BaseSchema):
    int_field: int
    _check_int_field = field_validator("int_field", is_large_enough)

    opt_int_field: Optional[int]
    _check_opt_int_field = optional_field_validator("opt_int_field", is_large_enough)

    field_to_parse: float = "3"  # Defaults also go through parsers!
    _parse_field = field_validator("field_to_parse", parse_float, pre=True)

    opt_field_to_parse: Optional[float]
    _parse_opt_field = field_validator("opt_field_to_parse", parse_float, pre=True)
    _check_opt_field = optional_field_validator("opt_field_to_parse", is_large_enough)  # multiple validators


def test_field_validator():
    DummySchema(int_field=7, opt_int_field=12, field_to_parse="3")
    DummySchema(int_field="8")  # auto-conversion must work

    assert DummySchema(int_field=8).field_to_parse == 8 + 3
    assert DummySchema(int_field=8, field_to_parse="5").field_to_parse == 8 + 5

    assert DummySchema(int_field=8, opt_field_to_parse="7").opt_field_to_parse == 8 + 7
    assert DummySchema(int_field=8, opt_field_to_parse=["a"]).opt_field_to_parse is None
    assert DummySchema(int_field=8, opt_field_to_parse=3).opt_field_to_parse is None
    assert DummySchema(int_field=8, opt_field_to_parse=None).opt_field_to_parse is None

    with pytest.raises(ValidationError):
        DummySchema(int_field=3)

    with pytest.raises(ValidationError):
        DummySchema(int_field="3")  # must fail with auto-conversion as well

    with pytest.raises(ValidationError):
        DummySchema(int_field=7, opt_int_field=3)

    with pytest.raises(ValidationError):
        DummySchema(int_field=7, field_to_parse="blabla")

    with pytest.raises(ValidationError):
        DummySchema(int_field=7, field_to_parse=2)

    with pytest.raises(ValidationError):
        DummySchema(int_field=7, opt_field_to_parse="-10")  # the second validator fails, the field is too small


def test_ensure_exactly_one_defined():
    class DummySchema(BaseSchema):
        param1: Optional[int]
        param2: Optional[float]
        _check_params = ensure_exactly_one_defined("param1", "param2")

    assert DummySchema(param1=0).param1 == 0
    assert DummySchema(param2=2.5).param2 == 2.5
    with pytest.raises(ValidationError):
        DummySchema()
    with pytest.raises(ValidationError):
        DummySchema(param1=0, param2=0)

    class DummySchema2(BaseSchema):
        param1: Optional[int] = 2
        param2: Optional[float]
        _check_params = ensure_exactly_one_defined("param1", "param2")

    assert DummySchema2().param1 == 2
    with pytest.raises(ValidationError):
        DummySchema2(param2=0)
    assert DummySchema2(param1=None, param2="3").param2 == 3


def test_ensure_defined_together():
    class DummySchema(BaseSchema):
        param1: Optional[int]
        param2: Optional[float]
        _check_params = ensure_defined_together("param1", "param2")

    schema1 = DummySchema()
    assert schema1.param1 is None and schema1.param2 is None
    schema2 = DummySchema(param1=2, param2=7.8)
    assert schema2.param1 == 2 and schema2.param2 == 7.8
    with pytest.raises(ValidationError):
        DummySchema()
        DummySchema(param1=0)
    with pytest.raises(ValidationError):
        DummySchema(param2=2.5)

    class DummySchema2(BaseSchema):
        param1: Optional[int] = 2
        param2: Optional[float]
        _check_params = ensure_defined_together("param1", "param2")

    schema3 = DummySchema2(param2="0.87")
    assert schema3.param1 == 2 and schema3.param2 == 0.87
    schema4 = DummySchema2(param1=None)
    assert schema4.param1 is None and schema4.param2 is None
    with pytest.raises(ValidationError):
        DummySchema2()


@pytest.mark.parametrize(
    "time_period,year,expected_start_date,expected_end_date",
    [
        ("yearly", 2020, "2020-01-01", "2020-12-31"),
        ("Q2", 2021, "2021-04-01", "2021-06-30"),
        ("Q2-yearly", 2021, "2020-07-01", "2021-06-30"),
    ],
)
def test_parse_time_period(time_period, year, expected_start_date, expected_end_date):
    start_date, end_date = parse_time_period([time_period, year])

    assert isinstance(start_date, dt.date)
    assert isinstance(end_date, dt.date)

    assert start_date.isoformat() == expected_start_date
    assert end_date.isoformat() == expected_end_date


@pytest.mark.parametrize("dtype_input", ["uint8", "float32", np.uint8, np.dtype("int16")])
def test_parse_dtype(dtype_input: Union[str, type, np.dtype]):
    class DtypeSchema(BaseSchema):
        dtype: np.dtype
        _parse_dtype = field_validator("dtype", parse_dtype, pre=True)

    schema = DtypeSchema(dtype=dtype_input)
    assert isinstance(schema.dtype, np.dtype)
    assert schema.dtype == np.dtype(dtype_input)


@pytest.mark.parametrize(
    "manager_input, succeeds",
    [
        ("eogrow.core.area.UtmZoneAreaManager", False),  # not a dict
        ({"wrong_field", "eogrow.core.area.UtmZoneAreaManager"}, False),
        ({"manager": "NonexistingManager"}, False),
        (
            {
                "manager": "eogrow.core.area.BatchAreaManager",
                "area": {"filename": "some_aoi.geojson"},
                "tiling_grid_id": 0,
                "resolution": 10,
            },
            True,
        ),
        (
            {
                "manager": "eogrow.core.logging.LoggingManager",
                "save_logs": True,
                "show_logs": True,
                "capture_warnings": False,
            },
            True,
        ),
        (
            {
                "manager": "eogrow.core.logging.LoggingManager",
                "save_logs": "yes please",
            },
            False,
        ),
        (
            {
                "manager": "eogrow.core.logging.LoggingManager",
                "extra_field": "should not be allowed",
            },
            False,
        ),
    ],
)
def test_validate_manager(manager_input: RawSchemaDict, succeeds: bool):
    class SchemaWithManager(BaseSchema):
        manager: ManagerSchema
        validate_manager = field_validator("manager", validate_manager, pre=True)

    with nullcontext() if succeeds else pytest.raises(ValidationError):
        SchemaWithManager(manager=manager_input)


@pytest.mark.parametrize(
    "collection_input, is_byoc, is_batch",
    [
        ("SENTINEL2_L2A", False, False),
        ("SENTINEL3_SLSTR", False, False),
        ("BYOC_blabla", True, False),
        ("BATCH_blabla", False, True),
    ],
)
def test_parse_collection_from_string(collection_input: str, is_byoc: bool, is_batch: bool):
    class CollectionSchema(BaseSchema):
        collection: DataCollection
        _parse_collection = field_validator("collection", parse_data_collection, pre=True)

    schema = CollectionSchema(collection=collection_input)
    assert isinstance(schema.collection, DataCollection)
    assert schema.collection.name in collection_input
    assert schema.collection.is_byoc == is_byoc
    assert schema.collection.is_batch == is_batch


def test_parse_collection_from_dict():
    class CollectionSchema(BaseSchema):
        collection: DataCollection
        _parse_collection = field_validator("collection", parse_data_collection, pre=True)

    raw_schema = {
        "name": "test",
        "api_id": "sentinel-2-l1c",
        "wfs_id": "DSS1",
        "service_url": "blabla",
        "processing_level": "L1C",
        "bands": [{"name": "Band1", "units": ["DN"], "output_types": ["float32"]}],
        "metabands": "SENTINEL2_L1C",
    }
    collection = CollectionSchema(collection=raw_schema).collection
    assert isinstance(collection, DataCollection)
    assert collection.bands == (Band("Band1", (Unit.DN,), (np.dtype("float32"),)),)
    assert collection.metabands is MetaBands.SENTINEL2_L1C
    assert collection.is_byoc is False

    raw_schema = {
        "name": "BYOC_test",
        "api_id": "override",
        "bands": [
            {"name": "Band1", "units": ["DN"], "output_types": ["float32"]},
            {"name": "Band2", "units": ["REFLECTANCE"], "output_types": ["bool"]},
        ],
    }
    collection = CollectionSchema(collection=raw_schema).collection
    assert isinstance(collection, DataCollection)
    assert collection.catalog_id == "byoc-test"
    assert collection.bands == (Band("Band1", (Unit.DN,), (np.float32,)), Band("Band2", (Unit.REFLECTANCE,), (bool,)))
    assert collection.is_byoc is True
