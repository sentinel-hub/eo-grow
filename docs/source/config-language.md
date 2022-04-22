## Config language

An important part of the `eo-grow` framework are configuration parameters which are kept separate from the code in a form of JSON files. In addition to the normal JSON syntax the framework implements a set of language rules defining how configuration parameters should be constructed and joined together.

| Language rule | Signature | Description | When is evaluated | Use cases |
|---|---|---|---|---|
| Config joins | A dictionary key that starts with `**` and points to a file path of another config, e.g. `{ ..., "**any_name": "path/to/another/config.json", ...}`. | Evaluation replaces the key with keys and values from the referenced config file. The replacement happens recursively. In case of clashes, parameters that already exist in a config have priority. The reason behind `**` notation is to be similar to `**kwargs` in Python. | When config is read from a file. | For referencing config files with parameters that are shared between pipelines. This rule aims to reduce config and parameter duplication. |
| Path to the config file | A dictionary value containing `${config_path}`, e.g. `"${config_path}/path/to/a/file"`. | The signature is a replaced with a path to the current config file. The path is relative to a filesystem and doesn't end with `/`. | When config is read from a file. | Can be used to reference another config file with a path that is relative to the current config location. |
| Path to a package/module | A dictionary value containing `${import_path:my_package.my_module}`, e.g. , `"${import_path:eogrow.pipelines}/path/to/a/file"`. | The signature is a replaced with a path to the location from where given import would be made. | At a pipeline initialization phase. | In case non-Python files are a part of a package, e.g. evalscripts, this provides their file path. |
| Reference a variable | A dictionary value containing `${var:my_variable}` and a subdictionary containing in form of `"variables": {"my_variable": "my_value", ...}` | The signature is replaced with values written in `variables` subdictionary and the subdictionary is removed in the process. | At a pipeline initialization phase. | This aims to reduce the number of duplicated or correlated config parameters and simplifies config parametrization. |
| Comments | `// A comment at the end of a line` or `/* A multi-line comment */` | The comments are ignored and removed when config is loaded. | When config is read from a file. | To explain why a parameter is set to a certain value. |

According to these rules there are `2` stages when rules are applied:

1. when config is read from a file,
   * This step is skipped in case configuration parameters are passed to a pipeline object as a dictionary in Python.
2. at a pipeline initialization phase,
   * In case configuration is passed to a remote instance this happens on the remote instance.

Additional notes:

- Dictionary keys must always be strings.
- Config language interpretation supports any nested combination of dictionaries and lists.
- Names of variables can only contain letters, numbers and `_`. Don't use `-`, `.` or any other characters.
- So far, config language is not completely OS-agnostic and it might not support Windows file paths.


### Pipeline joins

A typical configuration is a dictionary with pipeline parameters. However, it can also be a list of dictionaries. In this case each dictionary must contain parameters of a single pipeline. The order of dictionaries defines the consecutive order in which pipelines will be run. Example:

```
[
  {
    "pipeline": "FirstPipeline",
    "param1": "value1",
    ...
  },
  {
    "pipeline": "SecondPipeline",
    "param2": "value2",
    ...
  },
  ...
]
```
