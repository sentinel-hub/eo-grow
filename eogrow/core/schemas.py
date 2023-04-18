"""
Module defining a base pipeline schema and custom fields

For each pipeline a separate schema has to be defined which inherits from PipelineSchema. Such schema should be placed
as an internal class of the implemented pipeline class
"""
from inspect import isclass
from typing import List, Optional, Type, Union

from pydantic import BaseModel, Field
from pydantic.fields import ModelField

from ..types import BoolOrAuto, ImportPath
from ..utils.validators import field_validator, validate_manager
from .base import EOGrowObject

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


def build_schema_template(
    schema: Type[BaseModel],
    required_only: bool = False,
    pipeline_import_path: Optional[str] = None,
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


def _field_description(field: ModelField, description: Optional[str]) -> str:
    description = f" // {description}" if description else ""
    field_type = repr(field._type_display())
    default = repr(field.default) + " : " if field.default else ""
    return f"<< {default}{field_type}{description} >>"
