"""
Module defining a base pipeline schema and custom fields

For each pipeline a separate schema has to be defined which inherits from PipelineSchema. Such schema should be placed
as an internal class of the implemented pipeline class
"""

from __future__ import annotations

from inspect import isclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic.fields import ModelField

from ..types import ImportPath
from ..utils.validators import field_validator, validate_manager
from .base import EOGrowObject

# ruff: noqa: SLF001

BaseSchema = EOGrowObject.Schema


class ManagerSchema(BaseSchema):
    """A basic schema for managers, to be used as a parent class for defining manager schemas"""

    manager: Optional[ImportPath] = Field(description="An import path to this specific manager.")


class PipelineSchema(BaseSchema):
    """Base schema of the Pipeline class."""

    pipeline: Optional[ImportPath] = Field(description="Import path to an implementation of Pipeline class.")
    pipeline_name: Optional[str] = Field(
        description="Custom pipeline name for easier identification in logs. By default the class name is used."
    )

    storage: ManagerSchema = Field(description="A schema of an implementation of StorageManager class")
    validate_storage = field_validator("storage", validate_manager, pre=True)

    area: ManagerSchema = Field(description="A schema of an implementation of AreaManager class")
    validate_area = field_validator("area", validate_manager, pre=True)

    logging: ManagerSchema = Field(description="A schema of an implementation of LoggingManager class")
    validate_logging = field_validator("logging", validate_manager, pre=True)

    worker_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to ray tasks when executing via `RayExecutor`. The options are specified [here]"
            "(https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html)."
        ),
    )

    test_subset: Optional[List[Union[int, str]]] = Field(
        description=(
            "A list of EOPatch indices and/or names for which the pipeline is executed. Used for testing, can be set"
            " through CLI with the -t flag."
        )
    )
    skip_existing: bool = Field(
        False,
        description=(
            "Whether or not to skip already processed patches. In order to use this functionality a pipeline "
            "must implement the `filter_patch_list` method."
        ),
    )
    raise_on_temporal_mismatch: bool = Field(
        False, description="Treat `TemporalDimensionWarning` as an exception during EOExecution."
    )
    debug: bool = Field(False, description="Run pipeline without the `ray` wrapper to enable debugging.")


def build_schema_template(
    schema: type[BaseModel],
    required_only: bool = False,
    pipeline_import_path: str | None = None,
    add_descriptions: bool = False,
) -> dict:
    rec_flags: dict = dict(required_only=required_only, add_descriptions=add_descriptions)  # type is needed

    template: dict = {}
    for name, field in schema.__fields__.items():
        if required_only and not field.required:
            continue

        description = field.field_info.description if add_descriptions else None

        if name == "pipeline" and pipeline_import_path:
            template[name] = pipeline_import_path
        elif isclass(field.type_) and issubclass(field.type_, BaseModel):
            # Contains a subschema in the nesting
            if isclass(field.outer_type_) and issubclass(field.outer_type_, BaseModel):
                template[name] = build_schema_template(field.type_, **rec_flags)
                if description:
                    template[name]["<< description >>"] = description
            else:
                template[name] = {
                    "<< type >>": repr(field._type_display()),
                    "<< nested schema >>": str(field.type_),
                    "<< sub-template >>": build_schema_template(field.type_, **rec_flags),
                }
        else:
            template[name] = _field_description(field, description)

    return template


def _field_description(field: ModelField, description: str | None) -> str:
    description = f" // {description}" if description else ""
    field_type = repr(field._type_display())
    default = repr(field.default) + " : " if field.default else ""
    return f"<< {default}{field_type}{description} >>"
