# Common Configuration Patterns

## Using config templates

When you need to write a config for a pipeline, you can avoid rummaging through documentation by using the template command `eogrow-template`.

Invoking `eogrow-template "eogrow.pipelines.zipmap.ZipMapPipeline" "zipmap.json"` creates a file with the content:
```
{
  "pipeline": "eogrow.pipelines.zipmap.ZipMapPipeline",
  "pipeline_name": "<< Optional[str] >>",
  "workers": "<< 1 : int >>",
  "use_ray": "<< 'auto' : Union[Literal['auto'], bool] >>",
  "input_features": {
    "<< type >>": "List[InputFeatureSchema]",
    "<< nested schema >>": "<class 'eogrow.pipelines.zipmap.InputFeatureSchema'>",
    "<< sub-template >>": {
      "feature": "<< Tuple[FeatureType, str] >>",
      "folder_key": "<< str >>",
      "include_bbox_and_timestamp": "<< True : bool >>"
    }
  },
  ...
}
```
You can now remove any parameters you do not need and fill out the rest.

Parameter values are of form `"<< default : type >>"`, or `"<< default : type // description >>"` if you use the `--add-description` flag.

The parameters are in order of definition, causing `ZipMap` specific parameters come at the end (we switched the order a bit in the example).

In cases of nested schema, you get the output as the above for `"input_features"` which tells you what the type of the nesting is, and the template for the nested pydantic model.

For managers the template does not provide a schema directly, but the functionality is not restricted to pipelines, you can also invoke `eogrow-template "eogrow.core.logging.LoggingManager" "logging_manager.json"` to get templates for the logging manager.

## Global config

Most of the configuration files have a lot in common. This tends to be especially true for fields describing managers:
- `area`
- `storage`
- `logging`

From our experience, it is sometimes easiest to create a so-called *global configuration*, which contains all such fields.

```
{  // global_config.json
  "area": {
    ...
  },
  "storage": {
    ...
  },
  "logging": {
    ...
  }
}
```

This is then used in pipeline configurations.

```
{ // export.json
  "pipeline": "eogrow.pipelines.export_maps.ExportMapsPipeline",
  "**global_config": "${config_path}/global_config.json",
  "feature": ["data", "BANDS"],
  "map_dtype": "int16",
  "cogify": true,
  ...
}
```

This keeps pipeline configs shorter and more readable. One can also use multiple such files, for instance one for each manager. This makes it easy to have pipelines that work on different resolutions, where it's possible to just switch between `"**area_config": "${config_path}/area_10m.json"` and `"**area_config": "${config_path}/area_30m.json"`.

How fine-grained your config structure becomes is usually project-specific. Spreading it too thinly makes it harder to follow what precisely will be in the end config.

### Adjusting settings from the global config

In some cases, the settings from a global config (or from a different config file that you are importing) need to be overridden. Imagine that a pipeline produces a ton of useless warnings, and you only wish to ignore them for that specific pipeline.

```
{ // export.json
  "pipeline": "eogrow.pipelines.export_maps.ExportMapsPipeline",
  "**global_config": "${config_path}/global_config.json",
  "logging": {
    "capture_warnings": false
  },
  "feature": ["data", "BANDS"],
  "map_dtype": "int16",
  "cogify": true,
  ...
}
```

The processed configuration will have all the logging settings from `global_config.json`, except for `"capture_warnings"`. See [config language rules](config-language.html) for config joins.

## Pipeline chains

Pipeline chains are briefly touched in the config language docs, but only at the syntax level. Here we'll show two common usage patterns.

### End-to-end pipeline chain

In certain use cases we have multiple pipelines that are meant to be run in a certain succession. A great way of organizing that is via order-prefix naming, so `03_export_pipeline.json` is to be run as the third pipeline.

But the user still needs to run them in the correct order and by hand. This we can automate with a simple pipeline chain that links them together:
```
[ // end_to_end_run.json
  {"**download": "${config_path}/01_download.json"},
  {"**preprocess": "${config_path}/02_preprocess_data.json"},
  {"**predict": "${config_path}/03_use_model.json"},
  {"**export": "${config_path}/04_export_maps.json"},
  {"**ingest": "${config_path}/05_ingest_byoc.json"},
]
```

A simple `eogrow end_to_end_run.json` now runs all of these pipelines one after another.

### Rerunning with different parameters

In experimentation we often want to run the same pipeline for multiple parameter values. With a tiny bit of boilerplate this can also be taken care of with config chains.

```
[ // run_threshold_experiments.json
  {
    "variables": {"threshold": 0.1},
    "**pipeline": "${config_path}/extract_trees.json"
  },
  {
    "variables": {"threshold": 0.2},
    "**pipeline": "${config_path}/extract_trees.json"
  },
  {
    "variables": {"threshold": 0.3},
    "**pipeline": "${config_path}/extract_trees.json"
  },
  {
    "variables": {"threshold": 0.4},
    "**pipeline": "${config_path}/extract_trees.json"
  }
]
```

### Using variables with pipelines

While there is no syntactic sugar for specifying pipeline-chain-wide variables in JSON files, one can do that through CLI. Running `eogrow end_to_end_run.json -v "year:2019"` will set the variable `year` to 2019 for all pipelines in the chain.

## Path modification via variables

In some cases one wants fine grained control over path specifications. The following is a simplified example of how one can provide separate download paths for a large amount of batch pipelines.

```
{  // global_config.json
  "storage": {
    "structure": {
      "batch_tiffs": "batch-download/tiffs/year-${var:year}-${var:quarter}",
      ...
    },
    ...
  },
  ...
}
```

```
{ // batch_download.json
  "pipeline": "eogrow.pipelines.download_batch.BatchDownloadPipeline",
  "**global_config": "${config_path}/global_config.json",
  "output_folder_key": "batch_tiff",
  "inputs": [
    {
      "data_collection": "SENTINEL2_L2A",
      "time_period": "${var:year}-${var:quarter}"
    },
    ...
  ],
  ...
}
```

We now just need to provide the variables when running the config. This can be done either through the CLI via `eogrow batch_download.json -v "year:2019" -v "quarter:Q1"` or (for increased reproducibility) create configs with the variables specified in advance:

```
{ // batch_download_2019_Q4.json
  "**pipeline": "${config_path}/batch_download.json",
  "variables": {"year": 2019, "quarter": "Q4"}
}
```

In such cases, we advise you do not provide any variables in the core pipeline configuration (i.e. "batch_download.json") so that the config parsing fails if not all variables are specified. Otherwise you risk typo-specific problems such as specifying a value for `"yaer"` which won't override the `"year"` variable (and you download data for 2019 instead of 2020).

A similar specific-paths mechanism can also be achieved by modifying the storage manager directly from the final config:
```
{ // batch_download_2019_Q4.json
  "**pipeline": "${config_path}/batch_download.json",
  "variables": {"year": 2019, "quarter": "Q4"}
  "storage": {
    "structure": {
        "batch_tiffs": "batch-download/tiffs/year-2019-Q4"
    }
  }
}
```
While that is sufficient for many cases and more explicit, variables are preffered and might be less error-prone in case of complex folder structures.
