"""
Module defining common validators for schemas and validator wrappers
"""
import datetime as dt
import inspect
from typing import TYPE_CHECKING, Any, Callable, Tuple

from pydantic import validator

from sentinelhub import DataCollection

from .meta import collect_schema, import_object
from .types import S3Path, TimePeriod

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

    def optional_validator(value, values, config, field):  # type: ignore
        if value is not None:
            all_kwargs = {"values": values, "config": config, "field": field}
            kwargs = {k: v for k, v in all_kwargs.items() if k in additional_args}
            return validator_fun(value, **kwargs)
        return None

    return validator(field, allow_reuse=allow_reuse, **kwargs)(optional_validator)


def validate_s3_path(value: S3Path) -> S3Path:
    """Validates the prefix of a S3 bucket path"""
    assert value.startswith("s3://"), "S3 path must start with s3://"
    return value


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


def parse_data_collection(value: str) -> DataCollection:
    """Validates and parses data collection"""
    if value.startswith("BYOC_"):
        collection_id = value.split("_")[-1]
        return DataCollection.define_byoc(collection_id)

    if value.startswith("BATCH_"):
        collection_id = value.split("_")[-1]
        return DataCollection.define_batch(collection_id)

    if value in DataCollection.__members__:
        return getattr(DataCollection, value)
    raise ValueError(
        "Data collection should be a name of an existing DataCollection enum, 'BYOC_<collection_id>', "
        f"or 'BATCH_<collection id>' but name '{value}' was given."
    )


def validate_manager(value: dict) -> "ManagerSchema":
    """Parse and validate schema describing a manager."""
    assert "manager" in value, "Manager definition has no `manager` field that specifies its class."
    manager_class = import_object(value["manager"])
    manager_schema = collect_schema(manager_class)
    return manager_schema.parse_obj(value)  # type: ignore
