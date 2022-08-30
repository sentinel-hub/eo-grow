"""
Module defining common validators for schemas and validator wrappers
"""
import datetime as dt
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field, root_validator, validator

from sentinelhub import DataCollection
from sentinelhub.data_collections_bands import Band, Bands, MetaBands, Unit

from .meta import collect_schema, import_object
from .types import RawSchemaDict, S3Path, TimePeriod

if TYPE_CHECKING:
    from ..core.schemas import ManagerSchema


def field_validator(field: str, validator_fun: Callable, allow_reuse: bool = True, **kwargs: Any) -> classmethod:
    """Sugared syntax for the `validator` decorator of `pydantic`"""
    return validator(field, allow_reuse=allow_reuse, **kwargs)(validator_fun)


def optional_field_validator(
    field: str, validator_fun: Callable, allow_reuse: bool = True, **kwargs: Any
) -> classmethod:
    """Wraps the validator functions so that `None` is always a valid input and only calls the validator on values.

    This allows re-use of validators e.g. if we have a validator for `Path` we can now use it for `Optional[Path]`.
    Because `pydantic` has a variable amount of arguments passed to the validator this function can only be used
    with validators that include `**kwargs` (or require all three arguments). For details on this behaviour
    consult the [validators documentation](https://pydantic-docs.helpmanual.io/usage/validators/).
    """
    # In order to propagate the pydantic python magic we need a bit of python magic ourselves
    additional_args = inspect.getfullargspec(validator_fun).args[1:]

    def optional_validator(value, values, config, field):  # type: ignore[no-untyped-def]
        if value is not None:
            all_kwargs = {"values": values, "config": config, "field": field}
            kwargs = {k: v for k, v in all_kwargs.items() if k in additional_args}
            return validator_fun(value, **kwargs)
        return None

    optional_validator.__name__ = f"optional_{validator_fun.__name__}"  # used for docbuilding purposes
    # the correct way would be to use `functools.wraps` but this breaks pydantics python magic

    return validator(field, allow_reuse=allow_reuse, **kwargs)(optional_validator)


def validate_s3_path(value: S3Path) -> S3Path:
    """Validates the prefix of a S3 bucket path"""
    assert value.startswith("s3://"), "S3 path must start with s3://"
    return value


def ensure_exactly_one_defined(param1: str, param2: str, allow_reuse: bool = True, **kwargs: Any) -> classmethod:
    """A root validator that makes sure only one of the two parameters is defined."""

    def ensure_exclusion(cls: type, values: RawSchemaDict) -> RawSchemaDict:
        is_param1_defined = values.get(param1) is None
        is_param2_defined = values.get(param2) is None
        assert (
            is_param1_defined != is_param2_defined
        ), f"Exactly one of parameters `{param1}` and `{param2}` has to be specified."

        return values

    return root_validator(allow_reuse=allow_reuse, **kwargs)(ensure_exclusion)


def ensure_defined_together(param1: str, param2: str, allow_reuse: bool = True, **kwargs: Any) -> classmethod:
    """A root validator that makes sure that the two parameters are both (un)defined."""

    def ensure_exclusion(cls: type, values: RawSchemaDict) -> RawSchemaDict:
        is_param1_defined = values.get(param1) is None
        is_param2_defined = values.get(param2) is None
        assert (
            is_param1_defined == is_param2_defined
        ), f"Both or neither of parameters `{param1}` and `{param2}` have to be specified."

        return values

    return root_validator(allow_reuse=allow_reuse, **kwargs)(ensure_exclusion)


def parse_time_period(value: Tuple[str, str]) -> TimePeriod:
    """Allows parsing of preset options of shape `[preset_kind, year]` but that requires `pre` validation"""
    presets = ["yearly", "season", "Q1", "Q2", "Q3", "Q4", "Q1-yearly", "Q2-yearly", "Q3-yearly", "Q4-yearly"]

    if value[0] in presets:
        kind, year = value[0], int(value[1])
        start_dates = {
            "yearly": f"{year}-01-01",
            "season": f"{year-1}-09-01",
            "Q1": f"{year}-01-01",
            "Q2": f"{year}-04-01",
            "Q3": f"{year}-07-01",
            "Q4": f"{year}-10-01",
            "Q1-yearly": f"{year-1}-04-01",
            "Q2-yearly": f"{year-1}-07-01",
            "Q3-yearly": f"{year-1}-10-01",
            "Q4-yearly": f"{year}-01-01",
        }
        end_dates = {
            "yearly": f"{year}-12-31",
            "season": f"{year}-09-01",
            "Q1": f"{year}-03-31",
            "Q2": f"{year}-06-30",
            "Q3": f"{year}-09-30",
            "Q4": f"{year}-12-31",
            "Q1-yearly": f"{year}-03-31",
            "Q2-yearly": f"{year}-06-30",
            "Q3-yearly": f"{year}-09-30",
            "Q4-yearly": f"{year}-12-31",
        }
        value = start_dates[kind], end_dates[kind]

    start = dt.datetime.strptime(value[0], "%Y-%m-%d").date()
    end = dt.datetime.strptime(value[1], "%Y-%m-%d").date()
    assert start <= end, "Invalid start and end dates provided. End date must follow the start date"
    return start, end


def parse_dtype(value: str) -> np.dtype:
    return np.dtype(value)


def validate_manager(value: dict) -> "ManagerSchema":
    """Parse and validate schema describing a manager."""
    assert "manager" in value, "Manager definition has no `manager` field that specifies its class."
    manager_class = import_object(value["manager"])
    manager_schema = collect_schema(manager_class)
    return manager_schema.parse_obj(value)  # type: ignore[return-value]


class BandSchema(BaseModel):
    """Schema used in parsing DataCollection bands."""

    name: str
    units: Tuple[Unit, ...]
    output_types: Tuple[type, ...]

    @validator("output_types", pre=True, each_item=True)
    def _parse_output_types(cls, value: str) -> type:
        if value == "bool":
            return bool
        return np.dtype(value).type


class DataCollectionSchema(BaseModel):
    """Schema used in parsing DataCollection objects. Any extra parameters are passed to the definition as `**params`.
    """

    name: str = Field(
        "Name of the data collection. When defining BYOC collections use `BYOC_` prefix and for Batch collections use"
        " `BATCH_` to auto-generate fields with `define_byoc` or `define_batch`."
    )
    bands: Union[None, str, Tuple[BandSchema, ...]] = Field(
        None, description="Name of predefined collection in `Bands` or custom specification via `BandSchema`."
    )
    metabands: Union[None, str, Tuple[BandSchema, ...]] = Field(
        None, description="Name of predefined collection in `MetaBands` or custom specification via `BandSchema`."
    )

    class Config:
        extra = "allow"  # in order to pass on arbitrary parameters but keep definition shorter
        arbitrary_types_allowed = True


def _bands_parser(
    bands_collection: Union[Type[Bands], Type[MetaBands]], value: Union[None, str, Tuple[BandSchema, ...]]
) -> Optional[Tuple[Band, ...]]:
    """Collects defaults or parses bands from schemas."""

    if value is None:
        return None
    if isinstance(value, str):
        return getattr(bands_collection, value)
    return tuple(Band(band_schema.name, band_schema.units, band_schema.output_types) for band_schema in value)


def parse_data_collection(value: Union[str, dict, DataCollection]) -> DataCollection:
    """Validates and parses the data collection.

    If a string is given, then it tries to fetch a pre-defined collection. Otherwise it constructs a new collection
    according to the prefix of the name (`BYOC_` prefix to use `define_byoc` and `BATCH_` to use `define_batch`).
    """
    if isinstance(value, DataCollection):
        return value  # required in order to allow default values

    assert isinstance(value, (str, dict)), "Can only parse collection names or `DataCollectionSchema` definitions."

    params: Dict[str, Any] = {}
    if isinstance(value, str):
        name = value
        if name in DataCollection.__members__:
            return getattr(DataCollection, name)
    else:
        params = dict(DataCollectionSchema.parse_obj(value))
        params["bands"] = _bands_parser(Bands, params["bands"])
        params["metabands"] = _bands_parser(MetaBands, params["metabands"])
        name = params.pop("name")

    if name.startswith("BYOC_"):
        collection_id = name.split("_")[-1]
        return DataCollection.define_byoc(collection_id, **params)

    if name.startswith("BATCH_"):
        collection_id = name.split("_")[-1]
        return DataCollection.define_batch(collection_id, **params)

    return DataCollection.define(name, **params)
