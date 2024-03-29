# High Level Overview

The two main categories of `eo-grow` building blocks are:

- configurable objects (subclasses of `EOGrowObject`)
- configuration schemas (subclasses of `EOGrowObject.Schema`)

Each `EOGrowObject` is initialized with a `Schema` object. The `Schema` is saved to the object as an attribute `config: Schema` which stores the configuration information.

The configurable objects can be further separated into instances of:

- `Manager`, a helper class with a limited scope,
- `Pipeline`, a class for execution.

`Manager` classes are used to build configurations for specific aspects of the pipeline, such as area, storage, or logging, while the `Pipeline` class accepts the full configuration (pipeline specific + all managers) and contains methods of execution.

## Schemas

The `Schema` is in general a [pydantic model](https://docs.pydantic.dev/usage/models/), but with some project specific constrains and additions. It is best to always inherit from `EOGrowObject.Schema` to ensure a suitable pydantic configuration of the models.

The `EOGrow.Schema` model:

- rejects any additional parameters that are not listed,
- does not allow mutation,
- validates default values.

In case you are inheriting from a `Manger` or a `Pipeline` class, it is heavily advised to let `Schema` be a subclass of the superclass schema (type-checkers should warn you about it).

### Validators

You can use any kind of [pydantic validators](https://docs.pydantic.dev/usage/validators/) to verify the data in your schema. You can find some existing utility functions in `eogrow.utils.validators`:

- `field_validator` / `optional_field_validator` for wrapping callables defined elsewhere,
- `ensure_exactly_one_defined` and `ensure_defined_together` for linking together parameters that can be `None`,
- `ensure_storage_key_presence` for checking that storage keys are defined in the storage manager (see section on [managers](#managers)),
- `restrict_types` to restrict which feature types are allowed on a field that defines a feature.

Root validators can also be used, but are discouraged in the main `eo-grow` repository as they clutter the documentation pages.

For example, a storage key presence could be validated in the following way:

```python
class Schema(Pipeline.Schema):
    folder_key: str = "check_if_i_exist"
    _check_folder_key_presence = ensure_storage_key_presence("folder_key")
```

### Parsers

Certain types do not provide direct parsing capabilities (for instance `numpy.dtype` or `datetime`). In such cases you can use **pre-validators**, which means that the validator will be applied before `pydantic` checks that the type is right (check [here](https://docs.pydantic.dev/usage/validators/#pre-and-per-item-validators) for more info). This is done by setting the `pre` flag of validators to `True`. The `field_validator` and `optional_field_validator` utilities also allow this setting, so you can do:

```python
from eogrow.core.schemas import BaseSchema
from eogrow.utils.validators import optional_field_validator, parse_dtype

class MyModel(BaseSchema):
    maybe_dtype: Optional[np.dtype]
    _parse_maybe_dtype = optional_field_validator("maybe_dtype", parse_dtype, pre=True)
```

Other predefined parsers are and `parse_time_period` and `parse_data_collection`.

## Managers

Managers are helper-classes of pipelines that focus on a single role. Each manager is defined through a `Schema` configuration, where the configuration fields are specific to the manager at hand. This section focuses on different managers used by the `Pipeline` class and how to work with them. In the [pipelines](#pipelines) section we will then touch on how to connect all these managers to create and run a custom pipeline.

### Storage Manager

The storage manager takes care of data storage and works both with local storage and Amazon S3. It's primary purpose is to provide correct filesystem objects and filepaths in said filesystem. A basic overview of the `StorageManager` schema can be found below, for more information visit [the API docs](reference/eogrow.core.storage.html#eogrow.core.storage.StorageManager.Schema)

```json
{
  "manager": "eogrow.core.storage.StorageManager",
  "project_folder": "some_path/project_root_folder",
  "structure": {
    "data": "s2_downloaded_data",
    "reference": "reference",
    "models": "lgbm_models/built-up-detectors/models",
    "results": "built-up-predictions"
  }
}
```

To avoid keeping track of absolute paths, the storage manager utilizes a `key: path` mapping, which is specified as the `structure` parameter. Pipelines then operate with `input_folder_key="data"` instead of `input_path="some_path/project_root_folder/s2_downloaded_data"`. The approach is also much more resilient to typos.

Notable attributes/methods are:

- `filesystem` attribute, which can be used inside pipelines for IO.
- `get_folder` which, given a folder-key, provides the path in the `filesystem` to the desired folder.

While the folder-key approach appears limiting at first, it turns out to be flexible enough for the majority of cases. For more advanced use see [common configuration patterns](common-configuration-patterns.html).

### Area Managers

The Area Manager is a manager that takes care of how the area-of-interest (AOI) is loaded and how it is split into chunks to be processed by `eo-grow`. There are several pre-defined area managers available in the project, focusing on the few most common use cases for providing AOI specifications.

All area managers provide the following functionalities for development:

- `get_patch_list()`, for obtaining the list of patch names and corresponding bboxes
- `get_area_geometry()`, for obtaining the dissolved geometry of the AOI
- `get_grid(filtered = True|False)`, for obtaining the split AOI in the form of a grid

#### UTM Zone Area Manager

The `UtmZoneAreaManager` is probably the most commonly used area manager and most intuitive to work with. The user-provided geometry is split into patches of the user-provided size. If the AOI spans multiple UTM zones, the patches are grouped per zone. Read the [API docs](reference/eogrow.core.area.utm.html#eogrow.core.area.utm.UtmZoneAreaManager.Schema) on the `UTMZoneAreaManager` for more info.

#### Custom Grid Area Manager

For users which have a very specific way of splitting the AOI in mind, we provide the `CustomGridAreaManager`, which accepts a grid file of an already split AOI. The user only needs to provide the grid file folder key and name, along with the `name_column` parameter, which points to the column containing the patch names to be used. The folder key by default points to the `input_data` location, but could be any other location defined by the storage structure. Read the [API docs](reference/eogrow.core.area.custom_grid.html#eogrow.core.area.custom_grid.CustomGridAreaManager.Schema) on the `CustomGridAreaManager` for more info.

#### Batch Area Manager

For users working with [Sentinel Hub Batch API](https://docs.sentinel-hub.com/api/latest/api/batch/), we have prepared the `BatchAreaManager`, which splits the area according to [Sentinel Hub tiling grids](https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). This area manager is meant for larger projects focusing on larger areas.

The interface of the `BatchAreaManager` relies heavily on the predefined configuration options defined for the Batch API, so be sure to provide sensible values for the parameters. For example, the `tiling_grid_id` and `resolution` parameters should correspond to values stated in the [docs](https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids).

For existing projects involving Batch API, it is possible to provide the `batch_id` parameter, which will search for existing grids corresponding to the batch request. If the `batch_id` is not provided (this is by default), the `BatchAreaManager` will generate a new batch job with the given parameters. Read the [API docs](reference/eogrow.core.area.batch.html#eogrow.core.area.batch.BatchAreaManager.Schema) on the `BatchAreaManager` for more info.

### Logging Manager

The logging manager ensures that logging handlers are set up and configured correctly. It allows adjusting which packages to log to files, which to stdout, and which to ignore. It is unlikely you'll ever need to access any of it's methods directly. Use the standard `LOGGER = logging.getLogger(__name__)` for logging.

Settings that reference packages to ignore/show have a collection of default packages. One can reference them in the configuration with `"..."`.

```json
{
  "manager": "eogrow.core.logging.LoggingManager",
  "show_logs": true,
  "stdout_log_packages": ["...", "cool_package", "cooler_package"]
}
```

With the above settings the stdout logs will include `cool_package`, `cooler_package`, and also all of the default packages `eogrow`,
`__main__`, `root`, `sentinelhub.api.batch`.

## Pipelines

A `Pipeline` is an object focused towards executing a specific `EOWorkflow` over a collection of patches. It represents the interface for managing the data and logging with the use of [managers](#managers), as well as contains instructions for execution in the form of pipeline-specific tasks.

The `Pipeline` class has multiple _run_ methods that appear to have a similar functionality:

- `run` is the main execution method. It sets up logging and error handlers around `run_procedure`. _It is not meant to be changed._
- `run_procedure` contains instructions on what the pipeline does. By default, it creates a workflow with `build_workflow` and runs `run_execution`. _Override if you need the pipeline to also process things outside of an EOWorkflow (e.g. combine results)._
- `run_execution` takes care of logging and execution of the workflow. _It is not meant to be changed._
- `build_workflow` is a method that builds an `EOWorkflow` that the pipeline executes. _This is the method you usually want to implement._

In fact, when writing a custom pipeline, the majority of cases only need the following:

- defining the pipeline schema
- defining a custom `build_workflow` method
- constructing execution arguments (optional)
- providing filtering logic (optional)

The following sections expand on each item in the list above.

### Defining the Pipeline Schema

The configuration schema of the `Pipeline` class already has some pre-defined parameters which need to be provided for execution in addition to the managers. A full list of the parameters and their detailed descriptions can be found in the [pipeline schema API docs](reference/eogrow.core.pipeline.html#eogrow.core.pipeline.Pipeline.Schema).

Building a custom pipeline is straighforward, you only need to provide a suitable nested subclass of `Pipeline.Schema`, which must always be named `Schema`, as shown below:

```python
# example of how to write a custom pipeline
class MyPipeline(Pipeline):

    class Schema(Pipeline.Schema):
        extra_field: str = "beep"
        ...

    # this line informs type-checkers that the type of `config` is no longer `Pipeline.Schema`
    # but it is now `MyPipeline.Schema`
    config: Schema

    def custom_method():
        ...
    ...
```

### Building the Workflow

All pipelines expect an implementation of the `build_workflow` method, where the tasks for running specific work are defined and grouped into a workflow. Many workflows tend to be of the form:

1. Load patch
2. Perform specific tasks
3. Save patch and/or results

You can however load from multiple locations, merge patches, process and filter data, save some features and output some others. Anything you can do in `eo-learn` you can do here (but on a larger scale).

### Constructing Execution Arguments

In some cases, a task requires additional information at runtime, which can be unique per patch, such as the load/save location of a patch, or a specific bbox used to create a patch at the beginning of a pipeline.

By default, the method `get_execution_arguments` already configures execution arguments for `SaveTask`, `LoadTask`, and `CreateEOPatchTask` with the area manager data.

However, in cases where a custom task requires an extra parameter at runtime, it can be provided by updating the `get_execution_arguments` method of the `Pipeline` class. The method must set the arguments for each task which expects them, for all patches.

```python
def get_execution_arguments(self, workflow, patch_list):
    exec_kwargs = super().get_execution_arguments(workflow, patch_list)
    nodes = workflow.get_nodes()

    for patch_name, patch_args in exec_args.items():
        for node in nodes:
            if isinstance(node.task, MyCustomTask):
                patch_args[node] = my_custom_kwargs

    return exec_kwargs
```

### Providing Filtering Logic

Filtering logic is an optional part of the pipeline class and provides information on which patches to skip, in case they have already been processed. This is controlled via the `skip_existing` parameter in the [pipeline schema](#defining-the-pipeline-schema).

The filtering logic can be provided with the `filter_patch_list` method and depends very much on the what the user's definiton of "already processed" is. It could simply mean an existing patch directory in the storage, but it could depend on wheter some expected output is present or not.

Most commonly this boils down to checking for feature presence for all eopatches, and returning a list of patches where some/all features are missing. For this specific case we provide a utility method `get_patches_with_missing_features` under `eogrow.utils.filter`. The utility usage could look along the lines of:

```python
def filter_patch_list(self, patch_list):
    return get_patches_with_missing_features(filesystem, patches_folder, patch_list, features)
```

where the `features` parameter defines the list of features which must be present if the patch is to be skipped.
