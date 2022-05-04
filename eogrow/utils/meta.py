"""
Utilities for solving different problems in `eo-grow` package structure, which are mostly a pure Python magic.
"""
from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from ..core.pipeline import Pipeline
    from ..core.schemas import BaseSchema

_PIPELINE_PARAM_NAME = "pipeline"


def load_pipeline_class(config: dict) -> Type[Pipeline]:
    """Given a config object it loads the pipeline class referenced in the config"""
    pipeline_class_name = config.get(_PIPELINE_PARAM_NAME)
    if pipeline_class_name is None:
        raise ValueError(f"Config file is missing '{_PIPELINE_PARAM_NAME}' parameter, don't know which pipeline to use")

    pipeline_class = import_object(pipeline_class_name)
    return pipeline_class


def collect_schema(object_with_schema: Any) -> Type[BaseSchema]:
    """A utility that collects a schema from the given object.

    The object is expected to hold a unique internal class which inherits from `BaseSchema`. Example:

    class MyObject:
        class Schema(BaseSchema):
            ...

    This utility would provide `MySchema`. It works also if `MyObject` inherits from a class that holds the schema.
    """
    class_with_schema = object_with_schema if inspect.isclass(object_with_schema) else object_with_schema.__class__

    try:
        return class_with_schema.Schema
    except AttributeError as exception:
        raise SyntaxError(
            f"Class {class_with_schema} is missing a schema. Each EOGrowObject class needs to contain a pydantic "
            "model named `Schema`."
        ) from exception


def import_object(import_path: str) -> Any:
    """Imports an object from a given import path"""
    if "." not in import_path:
        raise ValueError(f"Import path {import_path} doesn't reference an object in a module.")
    module_name, object_name = import_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exception:
        raise ModuleNotFoundError(f"{exception}. Given import path '{import_path}' is invalid.") from exception

    if hasattr(module, object_name):
        return getattr(module, object_name)

    raise ImportError(
        f"Cannot import name '{object_name}' from {module_name} ({module.__file__}). Given import path "
        f"'{import_path}' is invalid."
    )


def get_os_import_path(import_path: str) -> str:
    """For a Python import path it provides OS import path.

    E.g. `eogrow.utils.meta` -> `/home/ubuntu/.../eogrow/utils/meta.py`
    """
    module_spec = importlib.util.find_spec(import_path)
    if module_spec is not None and module_spec.origin is not None:
        return module_spec.origin
    raise ValueError(f"Given import path {import_path} not found")


def get_package_versions() -> Dict[str, str]:
    """A utility function that provides dependency package versions

    At the moment it is and experimental utility. Everything is under try-catch in case something goes wrong

    :return: A dictionary with versions
    """
    try:
        import pkg_resources

        dependency_packages = ["eogrow"] + [
            requirement.name for requirement in pkg_resources.working_set.by_key["eogrow"].requires()  # type: ignore
        ]

        return {name: pkg_resources.get_distribution(name).version for name in dependency_packages}

    except BaseException as ex:
        return {"error": repr(ex)}
