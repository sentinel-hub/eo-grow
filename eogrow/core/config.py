"""
Module implementing management of configurations
"""
import base64
import binascii
import copy
import json
import pprint
import os
import re
from typing import Optional, Set, Dict, Callable, Tuple, Sequence, Union

import rapidjson
import fs.path
from munch import Munch, munchify, unmunchify
from eolearn.core.fs_utils import get_base_filesystem_and_path, join_path, get_full_path

from ..utils.general import jsonify
from ..utils.meta import get_os_import_path


class _BaseConfig:
    """A base class for configuration objects implementing methods that are common between `Config` and `ConfigList`"""

    @staticmethod
    def from_path(path: str) -> Union["Config", "ConfigList"]:
        """Loads from path in applies operations defined in `interpret_config_from_path`"""
        return interpret_config_from_path(path)

    @staticmethod
    def from_string(config_str: str) -> Union["Config", "ConfigList"]:
        """Creates a config object either from a base64-encoded string or a file path"""
        return get_config_from_string(config_str)

    def encode(self) -> str:
        """Dumps config into a json and the encodes it with base64

        :return: A base64-encoded string
        """
        json_string = json.dumps(self, default=jsonify)
        return base64.b64encode(json_string.encode()).decode()


class Config(Munch, _BaseConfig):
    """Class for handling `eo-grow` configurations

    This is a wrapper around `munch.Munch` and it allows to use dictionary parameters as attributes. E.g.
    besides `config['param']` you can also call `config.param`.

    Note that `Config(config_dict)` will only convert the top dictionary level into `Config`. Because of that the
    preferred way of loading is `Config.from_dict(config_dict)`.
    """

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Converts a normal dictionary into a `Config` object.

        The conversion is applied recursively on all nested dictionaries.
        """
        if not isinstance(config_dict, dict):
            raise ValueError(f"This method expected a dictionary, got {type(config_dict)}")
        return munchify(config_dict, factory=cls)

    def __repr__(self) -> str:
        """A nicer representation with pprint of the dictionary"""
        return f"{self.__class__.__name__}({{\n {pprint.pformat(dict(self))[1: -1]}\n}})"

    def __getattr__(self, item: str):
        """This additionally loads values of environmental variables

        This method performs the 3rd stage of language interpretation as described in
        `eo-grow/documentation/config-language.md`.
        """
        value = super().__getattr__(item)

        if isinstance(value, str):
            return _resolve_env_variables(value)

        if isinstance(value, list):
            return [(_resolve_env_variables(param) if isinstance(param, str) else param) for param in value]
        return value

    def add_cli_variables(self, cli_variables: Sequence[str]):
        """Adds variables passed in the CLI to the config. Has to be used before interpretation!"""
        cli_variable_mapping = dict(_parse_cli_variable(cli_var) for cli_var in cli_variables)
        current_variables = self.get("variables", {})
        current_variables.update(cli_variable_mapping)
        self["variables"] = current_variables

    def interpret(self) -> "Config":
        """Applies operations defined in `interpret_config_from_dict`"""
        return interpret_config_from_dict(self)

    def to_dict(self) -> dict:
        """Converts the object into a normal dictionary"""
        return unmunchify(self)


class ConfigList(list, _BaseConfig):
    """A class for handling a list of configurations"""

    @classmethod
    def from_list(cls, config_list: Sequence[dict]) -> "ConfigList":
        """Converts a normal list of dictionaries into a `ConfigList` object."""
        return cls(Config.from_dict(config) for config in config_list)

    def __repr__(self) -> str:
        """A nicer representation with pprint of the list"""
        return f"{self.__class__.__name__}([\n {pprint.pformat(list(self))[1: -1]}\n])"


def get_config_from_string(config_str: str) -> Union[Config, ConfigList]:
    """Provides a config object by either decoding a base64-encoded string or load it from a file path."""
    try:
        decoded_string = base64.b64decode(config_str.encode()).decode()
        raw_config = json.loads(decoded_string)

        if isinstance(raw_config, dict):
            return Config.from_dict(raw_config)
        return ConfigList.from_list(raw_config)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        pass

    return interpret_config_from_path(config_str)


def interpret_config_from_path(path: str, used_config_paths: Optional[Set[str]] = None) -> Union[Config, ConfigList]:
    """Loads and builds a config from parameters stored in files

    This function performs the 1st stage of language interpretation as described in
    eo-grow/documentation/config-language.md`.

    :param path: A full path where config is located
    :param used_config_paths: Just a helper parameter to prevent cyclic config imports.
    :return: A fully built config object.
    """
    filesystem, path = get_base_filesystem_and_path(path)
    with filesystem.open(path, "r") as file_handle:
        config = rapidjson.load(file_handle, parse_mode=rapidjson.PM_COMMENTS)

    used_config_paths = copy.copy(used_config_paths) or set()
    used_config_paths.add(get_full_path(filesystem, path))

    full_folder_path = get_full_path(filesystem, fs.path.dirname(path))
    config = _recursive_apply_to_strings(config, lambda value: _resolve_config_paths(value, full_folder_path))

    config = _recursive_config_build(config, used_config_paths)

    if isinstance(config, dict):
        return Config.from_dict(config)
    return ConfigList.from_list(config)


def _recursive_config_build(config, used_config_paths: Set[str]):
    """Recursively builds a configuration object by replacing dictionary items in form of

        `'**key': 'file path to another config'`

    with a content of a configuration file, referenced with the path. The items from both configuration objects are
    joined recursively.
    """
    if isinstance(config, dict):
        joint_config = {}
        imported_configs = []

        for key, value in config.items():
            if not isinstance(key, str):
                raise ValueError(f"Dictionary keys should always be strings, but found: {key}")

            if key.startswith("**"):
                if value in used_config_paths:
                    raise ValueError("Detected a cyclic import of configs")

                imported_config = interpret_config_from_path(value, used_config_paths=used_config_paths)
                imported_configs.append(imported_config)
            else:
                joint_config[key] = _recursive_config_build(value, used_config_paths)

        for imported_config in imported_configs:
            joint_config = recursive_config_join(joint_config, imported_config)

        return joint_config

    if isinstance(config, list):
        return [_recursive_config_build(value, used_config_paths) for value in config]

    return config


def interpret_config_from_dict(config: dict) -> Config:
    """Applies config language rules to a loaded config

    This function performs the 2nd stage of language interpretation as described in
    `eo-grow/documentation/config-language.md`.
    """
    _recursive_check_config(config)
    config = _recursive_apply_to_strings(config, _resolve_import_paths)

    variable_mapping = config.pop("variables", {})

    for variable_name in variable_mapping:
        if not re.fullmatch(r"\w+", variable_name):
            raise ValueError(f"Variable name {variable_name} contains illegal characters")

    config = _recursive_apply_to_strings(config, lambda config_str: _resolve_variables(config_str, variable_mapping))

    return Config.from_dict(config)


def _resolve_config_paths(config_str: str, config_path: str) -> str:
    """Replaces `${config_path}` with an actual path"""
    new_config_str = re.sub(r"\${config_path}", config_path, config_str)

    if new_config_str != config_str:
        return join_path(new_config_str)
    return new_config_str


def _resolve_import_paths(config_str: str) -> str:
    """Replaces `${import_path:package.module}` with an actual local path"""
    new_config_str = re.sub(r"\${import_path:([\w.]+)}", _sub_import_path, config_str)

    if new_config_str != config_str:
        return join_path(new_config_str)
    return new_config_str


def _sub_import_path(match: re.Match) -> str:
    """Substitutes a regex match with a local file path to a module or a package"""
    import_module = match.group(1)
    return os.path.dirname(get_os_import_path(import_module))


def _resolve_variables(config_str: str, variable_mapping: Dict[str, str]) -> str:
    """Replaces `${var:variable_name}` with a replacement value defined under variables"""
    return re.sub(r"\${var:(\w+)}", lambda match: _sub_variable(match, variable_mapping), config_str)


def _sub_variable(match: re.Match, variable_mapping: Dict[str, str]) -> str:
    """Substitutes a regex match with a new variable according to the mapping"""
    variable_name = match.group(1)
    if variable_name in variable_mapping:
        return str(variable_mapping[variable_name])
    raise ValueError(f"Variable name '{variable_name}' doesn't exist in a config dictionary of variables.")


def _resolve_env_variables(config_str: str) -> str:
    """Replaces `${env:variable_name}` with an environmental variable"""
    return re.sub(r"\${env:(\w+)}", _sub_env_variable, config_str)


def _sub_env_variable(match: re.Match) -> str:
    """Substitutes a regex match with a value set under environmental variable"""
    env_variable_name = match.group(1)
    env_var_value = os.getenv(env_variable_name)
    if env_var_value is None:
        raise KeyError(f"Environmental variable {env_variable_name} is not present in environment")
    return env_var_value


def _recursive_apply_to_strings(config, function: Callable):
    """Recursively applies a function on all string values (and not keys) of a nested config object"""
    if isinstance(config, dict):
        return {key: _recursive_apply_to_strings(value, function) for key, value in config.items()}

    if isinstance(config, list):
        return [_recursive_apply_to_strings(value, function) for value in config]

    if isinstance(config, str):
        return function(config)
    return config


def _recursive_check_config(config: dict):
    """Recursively checks if a config object conforms to some basic rules

    :raises: ValueError
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if not isinstance(key, str):
                raise ValueError(f"Config keys should be strings but {key} found")
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


def _parse_cli_variable(mapping_str: str) -> Tuple[str, str]:
    """Checks that the input is of shape `name:value` and then splits it into a tuple"""
    match = re.match(r"(?P<name>.+?):(?P<value>.+)", mapping_str)
    if match is None:
        raise ValueError(f'CLI variable input {mapping_str} is not of form `"name:value"`')
    parsed = match.groupdict()
    return parsed["name"], parsed["value"]
