"""
Module defining a base pipeline schema and custom fields

For each pipeline a separate schema has to be defined which inherits from PipelineSchema. Such schema should be placed
as an internal class of the implemented pipeline class
"""
from inspect import isclass
from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field
from pydantic.fields import ModelField

from ..utils.types import ImportPath, Path
from ..utils.validators import field_validator, validate_manager
from .base import EOGrowObject

BaseSchema = EOGrowObject.Schema


class ManagerSchema(EOGrowObject.Schema):
    """A basic schema for managers, to be used as a parent class for defining manager schemas"""

    manager: Optional[ImportPath] = Field(description="An import path to this specific manager.")


class LoggingManagerSchema(ManagerSchema):
    """Base schema of a logging manager. Only assumes fields required by the Pipeline class."""

    save_logs: bool = Field(
        False,
        description=(
            "A flag to determine if pipeline logs and reports will be saved to "
            "logs folder. This includes potential EOExecution reports and logs."
        ),
    )
    include_logs_to_report: bool = Field(
        False,
        description=(
            "If log files should be parsed into an EOExecution report file or just linked. When working "
            "with larger number of EOPatches the recommended option is False."
        ),
    )
    eoexecution_ignore_packages: Optional[List[str]] = Field(
        description=(
            "Names of packages which logs will not be written to EOExecution log files. The default null value "
            "means that a default list of packages will be used."
        )
    )


class PipelineSchema(EOGrowObject.Schema):
    """Base schema of the Pipeline class."""

    pipeline: Optional[ImportPath] = Field(description="Import path to an implementation of Pipeline class.")

    storage: ManagerSchema = Field(description="A schema of an implementation of StorageManager class")
    validate_storage = field_validator("storage", validate_manager, pre=True)

    area: ManagerSchema = Field(description="A schema of an implementation of AreaManager class")
    validate_area = field_validator("area", validate_manager, pre=True)

    eopatch: ManagerSchema = Field(description="A schema of an implementation of EOPatchManager class")
    validate_eopatch = field_validator("eopatch", validate_manager, pre=True)

    logging: LoggingManagerSchema = Field(description="A schema of an implementation of LoggingManager class")
    validate_logging = field_validator("logging", validate_manager, pre=True)

    workers: int = Field(
        1, description="Number of workers for parallel execution of workflows. Parameter does not affect ray clusters."
    )
    use_ray: Union[Literal["auto"], bool] = Field(
        "auto",
        description=(
            "Whether to run the pipeline locally or using a (local or remote) ray cluster. When using `auto` the"
            " pipeline checks if it can connect to a cluster, and if none are available runs locally."
        ),
    )

    patch_list: Optional[List[int]] = Field(
        description="A list of EOPatch indices for which the pipeline should be executed"
    )
    input_patch_file: Optional[Path] = Field(description="File path to a file with input EOPatch names")
    skip_existing: bool = Field(
        False,
        description=(
            "Whether or not to skip already processed patches. In order to use this functionality a pipeline "
            "must implement the `filter_patch_list` method."
        ),
    )


def build_schema_template(schema: Type[BaseModel]) -> dict:
    """From a given schema class it creates a template of a config file. It does that by modifying
    OpenAPI-style schema.
    """
    openapi_schema = schema.schema()

    template_mapping: dict = {}
    model_schemas = openapi_schema.get("definitions", {})
    for name, model_schema in model_schemas.items():
        model_template = _process_model_schema(model_schema, template_mapping)
        template_mapping[name] = model_template

    return _process_model_schema(openapi_schema, template_mapping)


def _process_model_schema(openapi_schema: dict, template_mapping: Dict[str, dict]) -> dict:
    """Processes schema for a single model into a template"""
    template = _get_basic_template(openapi_schema)

    params = openapi_schema.get("properties", {})
    required_params = set(openapi_schema.get("required", []))

    for param_name, param_schema in params.items():
        param_template = _get_basic_template(param_schema)

        if param_name in required_params:
            param_template["#required"] = True

        referred_model_name = _get_referred_model_name(param_schema)
        if referred_model_name:
            # In case param_template and referred_model have some parameters in common the following prioritizes the
            # ones from param_template
            for model_param_name, model_param_value in template_mapping[referred_model_name].items():
                if model_param_name not in param_template:
                    param_template[model_param_name] = model_param_value

        template[param_name] = param_template

    return template


_SUPPORTED_OPENAPI_FIELDS = {"title", "properties", "required", "definitions", "$ref", "allOf"}


def _get_basic_template(openapi_schema: dict) -> dict:
    """For an OpenAPI parameter schema it prepares a template with basic fields"""
    template = {}
    for key, value in openapi_schema.items():  # get metadata fields
        if key in _SUPPORTED_OPENAPI_FIELDS or (key == "type" and value == "object"):
            continue

        template[f"#{key}"] = value

    return template


def _get_referred_model_name(openapi_schema: dict) -> Optional[str]:
    """In a parameter schema it finds a reference to another model schema. If it doesn't exist it returns None"""
    referred_path = openapi_schema.get("$ref")
    if not referred_path:
        schema_items = openapi_schema.get("items", {})  # items can be a list
        referred_path = schema_items.get("$ref") if isinstance(schema_items, dict) else None

    if not referred_path and "allOf" in openapi_schema:
        referred_path = openapi_schema["allOf"][0].get("$ref")

    if referred_path:
        return referred_path.rsplit("/", 1)[-1]
    return referred_path


def build_minimal_template(
    schema: Type[BaseModel],
    required_only: bool,
    pipeline_import_path: Optional[str] = None,
    add_descriptions: bool = False,
) -> dict:
    rec_flags: dict = dict(required_only=required_only, add_descriptions=add_descriptions)  # type is needed
    json_schema = schema.schema()  # needed for descriptions

    template: dict = {}
    for name, field in schema.__fields__.items():

        if required_only and not field.required:
            continue

        if name == "pipeline" and pipeline_import_path:
            template[name] = pipeline_import_path
        elif isclass(field.type_) and issubclass(field.type_, BaseModel):
            # Contains a subschema in the nesting
            if isclass(field.outer_type_) and issubclass(field.outer_type_, BaseModel):
                template[name] = build_minimal_template(field.type_, **rec_flags)
                if "description" in json_schema["properties"][name]:
                    template[name]["<< description >>"] = json_schema["properties"][name]["description"]
            else:
                template[name] = {
                    "<< type >>": repr(field._type_display()),
                    "<< nested schema >>": str(field.type_),
                    "<< sub-template >>": build_minimal_template(field.type_, **rec_flags),
                }
        else:
            template[name] = _field_description(field, json_schema["properties"][name], add_descriptions)

    return template


def _field_description(field: ModelField, field_schema: dict, add_descriptions: bool) -> str:
    description = " // " + field_schema["description"] if "description" in field_schema and add_descriptions else ""
    field_type = repr(field._type_display())
    default = repr(field.default) + " : " if field.default else ""
    return f"<< {default}{field_type}{description} >>"
