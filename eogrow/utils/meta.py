"""
Utilities for solving different problems in `eo-grow` package structure, which are mostly a pure Python magic.
"""

from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING, Any

from packaging.requirements import Requirement

if TYPE_CHECKING:
    from ..core.pipeline import Pipeline
    from ..core.schemas import BaseSchema

_PIPELINE_PARAM_NAME = "pipeline"


def load_pipeline_class(config: dict) -> type[Pipeline]:
    """Given a config object it loads the pipeline class referenced in the config"""
    pipeline_class_name = config.get(_PIPELINE_PARAM_NAME)
    if pipeline_class_name is None:
        raise ValueError(f"Config file is missing `{_PIPELINE_PARAM_NAME}` parameter, don't know which pipeline to use")

    return import_object(pipeline_class_name)


def collect_schema(class_with_schema: type) -> type[BaseSchema]:
    """A utility that collects a schema from the given object.

    The object is expected to hold a unique internal class which inherits from `BaseSchema`. Example:

    class MyObject:
        class Schema(BaseSchema):
            ...

    This utility would provide `MySchema`. It works also if `MyObject` inherits from a class that holds the schema.
    """
    try:
        return class_with_schema.Schema  # type: ignore[attr-defined]
    except AttributeError as exception:
        raise SyntaxError(
            f"Class {class_with_schema} is missing a schema. Each `EOGrowObject` class needs to contain a pydantic "
            "model named `Schema`."
        ) from exception


def import_object(import_path: str) -> Any:
    """Imports an object from a given import path"""
    if "." not in import_path:
        raise ValueError(f"Import path `{import_path}` doesn't reference an object in a module.")
    module_name, object_name = import_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exception:
        raise ModuleNotFoundError(f"{exception}. Given import path `{import_path}` is invalid.") from exception

    if hasattr(module, object_name):
        return getattr(module, object_name)

    raise ImportError(
        f"Cannot import name `{object_name}` from {module_name} ({module.__file__}). Given import path "
        f"`{import_path}` is invalid."
    )


def get_package_versions() -> dict[str, str]:
    """A utility function that provides dependency package versions

    :return: A dictionary with versions
    """

    def is_base(req: str) -> bool:
        """Filters out packages needed only for development or documentation"""
        return re.search(r"extra == .(DEV|dev|DOCS|docs).", req) is None

    try:
        eogrow_reqs = importlib.metadata.requires("eo-grow") or []
        requirements = [Requirement(req).name for req in eogrow_reqs if is_base(req)]
        return {pkg: importlib.metadata.version(pkg) for pkg in ["eo-grow", *requirements]}

    except BaseException as ex:
        return {"error": repr(ex)}
