""" A module for testing eogrow.config
"""
import json
import os

import pytest
from fs.errors import ResourceNotFound

from eogrow.core.config import (
    Config,
    collect_configs_from_path,
    decode_config_list,
    encode_config_list,
    interpret_config_from_dict,
)
from eogrow.utils.meta import get_os_import_path

pytestmark = pytest.mark.fast


CONFIG_DICT = {
    "new_param": 4,
    "project_folder": "new_path",
    "nested_param": {"list_param": [1, 2, {"value": 3}], "foo": {"bar": 42}},
}
CONFIG_LIST = [CONFIG_DICT, CONFIG_DICT]


def test_config_repr():
    config = Config.from_dict(CONFIG_DICT)

    config_repr = repr(config)
    assert config_repr.startswith("Config")
    assert eval(config_repr) == config


def test_config_from_and_to_dict():
    config = Config.from_dict(CONFIG_DICT)

    assert isinstance(config, Config)
    assert config.new_param == 4
    assert isinstance(config.nested_param.list_param[2], Config)
    assert config.nested_param.list_param[2].value == 3
    assert config.nested_param.foo.bar == 42

    assert config == CONFIG_DICT  # Only compares values and doesn't distinguish between Config and dict instances

    config_dict = config.to_dict()
    assert config_dict == CONFIG_DICT
    assert not isinstance(config_dict, Config)
    assert not isinstance(config_dict["nested_param"]["list_param"][2], Config)


@pytest.mark.parametrize(
    "config_list, expected_size",
    [
        ([CONFIG_DICT], 164),
        (CONFIG_LIST, 328),
    ],
)
def test_config_encode_and_encode(config_list, expected_size):
    interpreted_configs = list(map(Config.from_dict, config_list))

    config_str = encode_config_list(interpreted_configs)
    assert isinstance(config_str, str)
    assert len(config_str) == expected_size

    decoded_configs = decode_config_list(config_str)
    assert isinstance(decoded_configs, list)
    assert all(isinstance(config, dict) for config in decoded_configs)

    assert decoded_configs == interpreted_configs


@pytest.mark.parametrize("config_object", [CONFIG_DICT, CONFIG_LIST])
def test_config_from_file(config_object, temp_folder):
    path = os.path.join(temp_folder, "config.json")
    with open(path, "w") as fp:
        json.dump(config_object, fp)

    config_list = list(map(interpret_config_from_dict, collect_configs_from_path(path)))
    if isinstance(config_object, dict):
        method_loaded_config = Config.from_path(path)
        assert len(config_list) == 1
        assert isinstance(method_loaded_config, Config) and isinstance(config_list[0], Config)
        assert method_loaded_config == config_object and config_list[0] == config_object

    else:
        assert isinstance(config_list, list) and all(isinstance(config, Config) for config in config_list)
        assert config_list == config_object


def test_missing_config_loading():
    false_config_filename = "./nonexistent-folder/foo.json"

    with pytest.raises(ResourceNotFound):
        Config.from_path(false_config_filename)


CONFIG_WITH_COMMENTS = """
// Some initial comment
{ //
  "key1": "value1", // key1
  "key2": [1, 2], /* key2 */
  "key3": {"a": "b"} /* ${config_file} */
}
/*
multi-line comment
*/
"""


def test_loading_with_comments(temp_folder):
    path = os.path.join(temp_folder, "config.json")
    with open(path, "w") as fp:
        fp.write(CONFIG_WITH_COMMENTS)

    config = Config.from_path(path)

    assert isinstance(config, Config)
    assert config.to_dict() == {"key1": "value1", "key2": [1, 2], "key3": {"a": "b"}}


CONFIG_WITH_CONFIG_PATH = {"path": "${config_path}/something", "not_path": "${}"}


def test_parsing_config_path(temp_folder):
    path = os.path.join(temp_folder, "config.json")
    with open(path, "w") as fp:
        json.dump(CONFIG_WITH_CONFIG_PATH, fp)

    config = Config.from_path(path)

    assert isinstance(config, Config)
    assert config.to_dict() == {"path": f"{temp_folder}/something", "not_path": "${}"}


CONFIG1 = {
    "**config2": "${config_path}/config2.json",
    "key": "value1",
    "list": [1, {"**config2": "${config_path}/config2.json"}],
}

CONFIG2 = {"key": "value2", "list": [1]}


def test_joining_configs(temp_folder):
    for index, config in enumerate([CONFIG1, CONFIG2]):
        path = os.path.join(temp_folder, f"config{index + 1}.json")
        with open(path, "w") as fp:
            json.dump(config, fp)

    path = os.path.join(temp_folder, "config1.json")
    config = Config.from_path(path)

    assert isinstance(config, Config)
    assert config.to_dict() == {"key": "value1", "list": [1, {"key": "value2", "list": [1]}]}


CONFIG3 = {"list": [{"**my-config": "${config_path}/config1.json"}]}


def test_cyclic_config_error(temp_folder):
    for index, config in enumerate([CONFIG1, CONFIG3]):
        path = os.path.join(temp_folder, f"config{index + 1}.json")
        with open(path, "w") as fp:
            json.dump(config, fp)

    for index in range(1, 3):
        path = os.path.join(temp_folder, f"config{index}.json")

        with pytest.raises(ValueError):
            Config.from_path(path)


CONFIG_WITH_IMPORT_PATHS = {"eogrow": "${import_path:eogrow}/xy", "other": ["${import_path:os.path}"]}


def test_parsing_import_paths():
    config = interpret_config_from_dict(CONFIG_WITH_IMPORT_PATHS)

    assert isinstance(config, Config)
    expected_dict = {
        "eogrow": os.path.dirname(get_os_import_path("eogrow")) + "/xy",
        "other": [os.path.dirname(get_os_import_path("os.path"))],
    }
    assert config.to_dict() == expected_dict


CONFIG_WITH_VARIABLES = {
    "variables": {"var1": "x", "var_2": 1, "3": "y"},
    "key": "${var:var1}, ${var:var_2}, ${var:3}, ${var:var1}",
}


def test_parsing_variables():
    config1 = interpret_config_from_dict(CONFIG_WITH_VARIABLES)
    config2 = Config.from_dict(CONFIG_WITH_VARIABLES)

    for config in [config1, config2]:
        assert isinstance(config, Config)
        assert config.to_dict() == {"key": "x, 1, y, x"}


def test_parsing_missing_variables():
    with pytest.raises(ValueError):
        Config.from_dict({"key": "${var:missing_var}"})


CONFIG_WITH_ENV_VARIABLES = {
    "key1": "${env:_TEST_ENV} -> ${env:_TEST_ENV2}",
    "key2": ["${env:_TEST_ENV}, ${env:_TEST_ENV2}"],
}


def test_parsing_env_variables():
    os.environ["_TEST_ENV"] = "xx"
    os.environ["_TEST_ENV2"] = "yy"

    config = interpret_config_from_dict(CONFIG_WITH_ENV_VARIABLES)

    assert isinstance(config, Config)
    assert config.to_dict() == CONFIG_WITH_ENV_VARIABLES

    assert config.key1 == "xx -> yy"
    assert config.key2 == ["xx, yy"]


def test_cli_variables():
    # Overwrite
    config = interpret_config_from_dict(CONFIG_WITH_VARIABLES, {"var1": 1, "var_2": "naughty ducks stole my grapes"})
    assert config.to_dict() == {"key": "1, naughty ducks stole my grapes, y, 1"}

    # Add
    raw_config = CONFIG_WITH_VARIABLES.copy()
    raw_config.pop("variables")
    assert "variables" not in raw_config

    config = interpret_config_from_dict(raw_config, {"var1": "1", "var_2": "2", "3": "4"})
    assert config.to_dict() == {"key": "1, 2, 4, 1"}
