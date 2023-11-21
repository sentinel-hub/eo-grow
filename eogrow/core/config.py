"""Implements functions that transform raw dictionaries/JSON files according to the config language of eo-grow."""

from __future__ import annotations

import copy
import re
from functools import reduce
from typing import Any, Callable, List, NewType, Union, cast

import fs.path
import rapidjson

from eolearn.core.utils.fs import get_base_filesystem_and_path, get_full_path, join_path

CrudeConfig = NewType("CrudeConfig", dict)
RawConfig = NewType("RawConfig", dict)


def collect_configs_from_path(path: str, used_config_paths: set[str] | None = None) -> CrudeConfig | list[CrudeConfig]:
    """Loads and builds a list of config dictionaries defined by the parameters stored in files

    This function performs the 1st stage of language interpretation as described in
    eo-grow/documentation/config-language.md`.

    :param path: A full path where a config file is located
    :param used_config_paths: A helper parameter to prevent cyclic config imports.
    :return: A list of stage 1 dictionaries from which to build config object.
    """
    filesystem, path = get_base_filesystem_and_path(path)
    with filesystem.open(path, "r") as file_handle:
        config = rapidjson.load(file_handle, parse_mode=rapidjson.PM_COMMENTS)

    used_config_paths = copy.copy(used_config_paths) or set()
    used_config_paths.add(get_full_path(filesystem, path))

    full_folder_path = get_full_path(filesystem, fs.path.dirname(path))
    config = _recursive_apply_to_strings(config, lambda value: _resolve_config_paths(value, full_folder_path))

    config = _recursive_config_build(config, used_config_paths)

    if not isinstance(config, (dict, list)):
        raise TypeError(f"When interpreting config from {path} a dictionary or list was expected, got {type(config)}.")
    return cast(Union[CrudeConfig, List[CrudeConfig]], config)


def _recursive_config_build(config: object, used_config_paths: set[str]) -> object:
    """Recursively builds a configuration object by replacing dictionary items in form of

        `'**key': 'file path to another config'`

    with a content of a configuration file, referenced with the path. The items from both configuration objects are
    joined recursively.
    """
    if isinstance(config, dict):
        joint_config = {}
        imported_configs: list[CrudeConfig] = []

        for key, value in config.items():
            if not isinstance(key, str):
                raise TypeError(f"Dictionary keys should always be strings, but found: {key}")

            if key.startswith("**"):
                if value in used_config_paths:
                    raise ValueError("Detected a cyclic import of configs")

                imported_config = collect_configs_from_path(value, used_config_paths=used_config_paths)
                if not isinstance(imported_config, dict):
                    raise ValueError(
                        "Config lists cannot be imported inside configs. Found a config list when resolving key"
                        f" {key} for path {value}"
                    )
                imported_configs.append(imported_config)
            else:
                joint_config[key] = _recursive_config_build(value, used_config_paths)

        return reduce(recursive_config_join, imported_configs, joint_config)

    if isinstance(config, list):
        return [_recursive_config_build(value, used_config_paths) for value in config]

    return config


def interpret_config_from_dict(config: CrudeConfig, external_variables: dict[str, Any] | None = None) -> RawConfig:
    """Applies config language rules to a loaded config

    This function performs the 2nd stage of language interpretation as described in
    `eo-grow/documentation/config-language.md`.
    """
    _recursive_check_config(config)

    if not isinstance(config, dict):
        raise TypeError(f"Can only interpret dictionary objects, got {type(config)}.")

    config = cast(CrudeConfig, config.copy())
    variable_mapping = config.pop("variables", {})
    if external_variables:
        variable_mapping.update(external_variables)

    for variable_name in variable_mapping:
        if not re.fullmatch(r"\w+", variable_name):
            raise ValueError(f"Variable name {variable_name} contains illegal characters")

    config_with_variables = _recursive_apply_to_strings(
        config, lambda config_str: _resolve_variables(config_str, variable_mapping)
    )

    return cast(RawConfig, config_with_variables)


def interpret_config_from_path(path: str) -> RawConfig:
    """Loads from path in applies both steps of the config language."""
    config = collect_configs_from_path(path)
    if isinstance(config, dict):
        return interpret_config_from_dict(config)
    raise ValueError(f"The JSON file {path} was expected to contain a single dictionary, got {len(config)}")


def _resolve_config_paths(config_str: str, config_path: str) -> str:
    """Replaces `${config_path}` with an actual path"""
    new_config_str = re.sub(r"\${config_path}", config_path, config_str)

    if new_config_str != config_str:
        return join_path(new_config_str)
    return new_config_str


def _resolve_variables(config_str: str, variable_mapping: dict[str, str]) -> str:
    """Replaces `${var:variable_name}` with a replacement value defined under variables"""
    return re.sub(r"\${var:(\w+)}", lambda match: _sub_variable(match, variable_mapping), config_str)


def _sub_variable(match: re.Match, variable_mapping: dict[str, str]) -> str:
    """Substitutes a regex match with a new variable according to the mapping"""
    variable_name = match.group(1)
    if variable_name in variable_mapping:
        return str(variable_mapping[variable_name])
    raise ValueError(f"Variable name `{variable_name}` doesn't exist in a config dictionary of variables.")


def _recursive_apply_to_strings(config: object, function: Callable) -> object:
    """Recursively applies a function on all string values (and not keys) of a nested config object"""
    if isinstance(config, dict):
        return {function(key): _recursive_apply_to_strings(value, function) for key, value in config.items()}

    if isinstance(config, list):
        return [_recursive_apply_to_strings(value, function) for value in config]

    if isinstance(config, str):
        return function(config)
    return config


def _recursive_check_config(config: object) -> None:
    """Recursively checks if the config satisfies basic conditions for being JSON serializable."""
    if isinstance(config, dict):
        for key, value in config.items():
            if not isinstance(key, str):
                raise TypeError(f"Config keys should be strings but {key} found")
            _recursive_check_config(value)

    elif isinstance(config, list):
        for value in config:
            _recursive_check_config(value)


def recursive_config_join(config1: dict, config2: dict) -> dict:
    """Recursively join 2 config objects, where config1 values override config2 values"""
    for key, value in config2.items():
        if key not in config1:
            config1[key] = value
        elif isinstance(config1[key], dict) and isinstance(value, dict):
            config1[key] = recursive_config_join(config1[key], value)

    return config1
