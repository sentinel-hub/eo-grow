import json
import os

import pytest
from fs.errors import ResourceNotFound

from eogrow.core.config import collect_configs_from_path, interpret_config_from_dict, interpret_config_from_path

CONFIG_DICT = {
    "new_param": 4,
    "project_folder": "new_path",
    "nested_param": {"list_param": [1, 2, {"value": 3}], "foo": {"bar": 42}},
}
CONFIG_LIST = [CONFIG_DICT, CONFIG_DICT]


def test_config_from_file_single(temp_folder):
    path = os.path.join(temp_folder, "config.json")
    with open(path, "w") as fp:
        json.dump(CONFIG_DICT, fp)

    directly_loaded_config = interpret_config_from_path(path)
    assert isinstance(directly_loaded_config, dict)
    assert directly_loaded_config == CONFIG_DICT
    assert directly_loaded_config == interpret_config_from_dict(collect_configs_from_path(path))


def test_config_from_file_chain(temp_folder):
    path = os.path.join(temp_folder, "config.json")
    with open(path, "w") as fp:
        json.dump(CONFIG_LIST, fp)

    config_list = collect_configs_from_path(path)
    assert isinstance(config_list, list)
    assert all(isinstance(config, dict) for config in config_list)
    assert config_list == CONFIG_LIST


def test_missing_config_loading():
    false_config_filename = "./nonexistent-folder/foo.json"

    with pytest.raises(ResourceNotFound):
        interpret_config_from_path(false_config_filename)


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

    config = interpret_config_from_path(path)

    assert config == {"key1": "value1", "key2": [1, 2], "key3": {"a": "b"}}


CONFIG_WITH_CONFIG_PATH = {"path": "${config_path}/something", "not_path": "${}"}


def test_parsing_config_path(temp_folder):
    path = os.path.join(temp_folder, "config.json")
    with open(path, "w") as fp:
        json.dump(CONFIG_WITH_CONFIG_PATH, fp)

    config = interpret_config_from_path(path)

    assert config == {"path": f"{temp_folder}/something", "not_path": "${}"}


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
    config = interpret_config_from_path(path)

    assert config == {"key": "value1", "list": [1, {"key": "value2", "list": [1]}]}


CONFIG3 = {"list": [{"**my-config": "${config_path}/config1.json"}]}


def test_cyclic_config_error(temp_folder):
    for index, config in enumerate([CONFIG1, CONFIG3]):
        path = os.path.join(temp_folder, f"config{index + 1}.json")
        with open(path, "w") as fp:
            json.dump(config, fp)

    for index in range(1, 3):
        path = os.path.join(temp_folder, f"config{index}.json")

        with pytest.raises(ValueError):
            interpret_config_from_path(path)


CONFIG_WITH_VARIABLES = {
    "variables": {"var1": "x", "var_2": 1, "3": "y"},
    "key": "${var:var1}, ${var:var_2}, ${var:3}, ${var:var1}",
}


def test_parsing_variables():
    config = interpret_config_from_dict(CONFIG_WITH_VARIABLES)
    assert config == {"key": "x, 1, y, x"}


def test_parsing_variables_in_dict_keys():
    config_with_variables_in_keys = {
        "variables": {"var1": "x", "var_2": 1},
        "${var:var1}": "${var:var1}, ${var:var_2}",
    }
    config = interpret_config_from_dict(config_with_variables_in_keys)
    assert config == {"x": "x, 1"}


def test_parsing_missing_variables():
    with pytest.raises(ValueError):
        interpret_config_from_dict({"key": "${var:missing_var}"})


def test_cli_variables():
    # Overwrite
    config = interpret_config_from_dict(CONFIG_WITH_VARIABLES, {"var1": 1, "var_2": "naughty ducks stole my grapes"})
    assert config == {"key": "1, naughty ducks stole my grapes, y, 1"}

    # Add
    raw_config = CONFIG_WITH_VARIABLES.copy()
    raw_config.pop("variables")
    assert "variables" not in raw_config

    config = interpret_config_from_dict(raw_config, {"var1": "1", "var_2": "2", "3": "4"})
    assert config == {"key": "1, 2, 4, 1"}
