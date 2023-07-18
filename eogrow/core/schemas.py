"""
Module defining a base pipeline schema and custom fields

For each pipeline a separate schema has to be defined which inherits from PipelineSchema. Such schema should be placed
as an internal class of the implemented pipeline class
"""
from __future__ import annotations

from inspect import isclass
from typing import Any, List, Optional, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ..types import BoolOrAuto, ImportPath
from ..utils.validators import field_validator, validate_manager
from .base import EOGrowObject

# ruff: noqa: SLF001

BaseSchema = EOGrowObject.Schema


class ManagerSchema(BaseSchema):
    """A basic schema for managers, to be used as a parent class for defining manager schemas"""

    manager: Optional[ImportPath] = Field(None, description="An import path to this specific manager.")


class PipelineSchema(BaseSchema):
    """Base schema of the Pipeline class."""

    pipeline: Optional[ImportPath] = Field(None, description="Import path to an implementation of Pipeline class.")
    pipeline_name: Optional[str] = Field(
        None, description="Custom pipeline name for easier identification in logs. By default the class name is used."
    )

    storage: ManagerSchema = Field(description="A schema of an implementation of StorageManager class")
    validate_storage = field_validator("storage", validate_manager, pre=True)

    area: ManagerSchema = Field(description="A schema of an implementation of AreaManager class")
    validate_area = field_validator("area", validate_manager, pre=True)

    logging: ManagerSchema = Field(description="A schema of an implementation of LoggingManager class")
    validate_logging = field_validator("logging", validate_manager, pre=True)

    workers: int = Field(
        1, description="Number of workers for parallel execution of workflows. Parameter does not affect ray clusters."
    )
    use_ray: BoolOrAuto = Field(
        "auto",
        description=(
            "Whether to run the pipeline locally or using a (local or remote) ray cluster. When using `auto` the"
            " pipeline checks if it can connect to a cluster, and if none are available runs locally."
        ),
    )

    test_subset: Optional[List[Union[int, str]]] = Field(
        None,
        description=(
            "A list of EOPatch indices and/or names for which the pipeline is executed. Used for testing, can be set"
            " through CLI with the -t flag."
        ),
    )
    skip_existing: bool = Field(
        False,
        description=(
            "Whether or not to skip already processed patches. In order to use this functionality a pipeline "
            "must implement the `filter_patch_list` method."
        ),
    )


def build_schema_template(
    schema: type[BaseModel],
    required_only: bool = False,
    pipeline_import_path: str | None = None,
    add_descriptions: bool = False,
) -> dict:
    rec_flags: dict = dict(required_only=required_only, add_descriptions=add_descriptions)  # type is needed

    template: dict = {}
    for name, field in schema.model_fields.items():
        if required_only and not field.is_required():
            continue

        # remove optional layer around type
        is_optional = _is_optional_type(field.annotation)
        field_type = field.annotation if not is_optional else _get_nested_type(field.annotation)

        # get inner type in case of nested fields (list, tuple, ..)
        origin_type = get_origin(field_type)
        core_type = field_type if origin_type is None else _get_nested_type(field_type)

        description = field.description if add_descriptions else None

        if name == "pipeline" and pipeline_import_path:
            template[name] = pipeline_import_path
        elif isclass(core_type) and issubclass(core_type, BaseModel):
            if origin_type is None:  # expand schema
                template[name] = build_schema_template(field_type, **rec_flags)
                if description:
                    template[name]["<< description >>"] = description
            else:  # expand the nested subschema
                template[name] = {
                    "<< type >>": _type_name(field_type),
                    "<< nested schema >>": repr(core_type),
                    "<< sub-template >>": build_schema_template(core_type, **rec_flags),
                }
        else:
            template[name] = _field_description(field, description)

    return template


def _type_name(field_type: Any) -> str:
    """Return the prettified class/type name."""
    if isclass(field_type):
        return field_type.__name__

    return repr(field_type).replace("typing.", "")


def _is_optional_type(field_type: Any) -> bool:
    """Returns True if type of field is Optional, otherwise False."""
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def _get_nested_type(field_type: Any) -> Any:
    """Return the first nested type. type(None) is ignored"""
    return next(item for item in get_args(field_type) if item is not type(None))


def _field_description(field: FieldInfo, description: str | None) -> str:
    description = f" // {description}" if description else ""
    default = repr(field.default) + ": " if field.default not in [None, PydanticUndefined] else ""
    return f"<< {default}{_type_name(field.annotation)}{description} >>"
