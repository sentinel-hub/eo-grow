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

The `Schema` is in general a [`pydantic` model](https://docs.pydantic.dev/usage/models/), but with some project specific constrains and additions. It is best to always inherit from `EOGrowObject.Schema` to ensure a suitable pydantic configuration of the models.

The `EOGrow.Schema` model:

- rejects any additional parameters that are not listed,
- does not allow mutation,
- validates default values.

In case you are inheriting from a `Manger` or a `Pipeline` class, it is heavily advised to let `Schema` be a subclass of the superclass schema (type-checkers should warn you about it).

### Validators

You can use any kind of [`pydantic` validators](https://docs.pydantic.dev/usage/validators/) to verify the data in your schema. You can find some existing utility functions in `eogrow.utils.validators`:

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

The storage manager takes care of data storage and works both with local storage and Amazon S3. It's primary purpose is to provide correct filesystem objects and filepaths in said filesystem.

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

The `UtmZoneAreaManager` is probably the most commonly used area manager and most intuitive to work with. The user-provided geometry is split into patches of the user-provided size. If the AOI spans multiple UTM zones, the patches are grouped per zone. Here is what the user-provided configuration looks like:

```json
{
  "geometry_filename": <str>
  "patch": {
    "size_x": <int>,
    "size_y": <int>,
    "buffer_x": <float>,
    "buffer_y": <float>
  },
  "offset_x": <float>,
  "offset_y": <float>
}
```

#### Custom Grid Area Manager

For users which have a very specific way of splitting the AOI in mind, we provide the `CustomGridAreaManager`, which accepts a grid file of an already split AOI. The user only needs to provide the grid file folder key and name, along with the `name_column` parameter, which points to the column containing the patch names to be used. The folder key by default points to the `input_data` location, but could be any other location defined by the storage structure.

```json
{
  "grid_folder_key": <str>,
  "grid_filename": <str>,
  "name_column": <str>
}
```

#### Batch Area Manager

For users working with [Sentinel Hub Batch API](https://docs.sentinel-hub.com/api/latest/api/batch/), we have prepared the `BatchAreaManager`, which splits the area according to [Sentinel Hub tiling grids](https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). This area manager is meant for larger projects focusing on larger areas.

The interface of the `BatchAreaManager` relies heavily on the predefined configuration options defined for the Batch API, so be sure to provide sensible values for the parameters. For example, the `tiling_grid_id` and `resolution` parameters should correspond to values stated in the [docs](https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids).

For existing projects involving Batch API, it is possible to provide the `batch_id` parameter, which will search for existing grids corresponding to the batch request. If the `batch_id` is not provided (this is by default), the `BatchAreaManager` will generate a new batch job with the given parameters.

```json
{
  "geometry_filename": <str>,
  "tiling_grid_id": <int>,
  "resolution": <float>,
  "tile_buffer_x": <int>,
  "tile_buffer_y": <int>,
  "batch_id": <str|None>
}
```

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

A `Pipeline` is an object focused towards executing a specific workflow over a collection of patches. It represents the interface for managing the data and logging with the use of [managers](#managers), as well as contains instructions for execution in the form of pipeline-specific tasks.

The basic parts of writing a custom `Pipeline` class usually consist of the following:

- defining the pipeline schema
- building the workflow
- constructing execution arguments (optional)
- providing filtering logic (optional)

The following sections expand on each item in the list above.

### Defining the Pipeline Schema

The configuration schema of the `Pipeline` class already has some pre-defined parameters which need to be provided for execution. Besides the expected area, storage, and logging manager schemas, these are:

```json
{
  "pipeline": <str: import path to an implementation of the Pipeline class>,
  "pipeline_name": <str|None: custom pipeline name>,
  "workers": <int: number of workers for parallel execution of workflows>,
  "use_ray": <bool|None: run the pipeline locally or on A ray cluster>,
  "test_subset": <list(int)|list(str)|None: a list of patch indices and/or names for execution>,
  "skip_existing": <bool: skip already processed patches>,

  "area": <AreaManager>,
  "storage": <StorageManager>,
  "logging": <LoggingManager>
}
```

For a more detailed description of the parameters, you can [read the API docs](https://eo-grow.readthedocs.io/en/latest/reference/eogrow.core.pipeline.html#eogrow.core.pipeline.Pipeline.Schema).

Building a custom object as a subclass of `EOGrowObject` is straighforward, you only need to provide a suitable nested subclass of `EOGrowObject.Schema`, which must always be named `Schema`. For example, a subclass of `Pipeline` should contain a nested subclass of `Pipeline.Schema`, as shown below.

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

All pipelines expect an implementation of the `build_workflow` method, where the tasks for running specific work are defined and grouped into a workflow. Usually this consists of (but is not limited to) simple, linearly connected tasks, usually in the form of:

1. Load patch
2. Perform specific task
3. Save patch and/or results

### Constructing Execution Arguments

In some cases, a task requires additional information at runtime, which can be unique per patch, such as the load/save location of a patch, or a specific bbox used to create a patch at the beginning of a pipeline.

For the specific examples mentioned above, the execution argument building step have already been implemented, meaning that for simple pipelines with simple tasks this particular step is optional.

However, in cases where a custom task requires an extra parameter at runtime, it can be provided by updating the `get_execution_arguments()` method of the `Pipeline` class. The method must set the arguments for each task which expects them, for all patches.

```python
def get_execution_arguments(self, workflow, patch_list):
    exec_kwargs = {}
    nodes = workflow.get_nodes()
    for name, bbox in patch_list:
        patch_args = {}

        for node in nodes:
            if isinstance(node.task, MyCustomTask):
                patch_args[node] = dict(**custom_kwargs)

        exec_kwargs[name] = patch_args
    return exec_kwargs
```

### Providing Filtering Logic

Filtering logic is an optional part of the pipeline class and provides information on which patches to skip, in case they have already been processed. This is controlled via the `skip_existing` parameter in the [pipeline schema](#defining-the-pipeline-schema).

The filtering logic can be provided with the `filter_patch_list()` method and depends very much on the what the user's definiton of "already processed" is. It could simply mean an existing patch directory in the storage, but it could depend on wheter some expected output is present or not.

Most commonly this boils down to checking for feature presence for all eopatches, and returning a list of patches where some/all features are missing. For this specific case we provide a utility method `get_patches_with_missing_features()` under `eogrow.utils.fitler. The utility usage could look along the lines of:

```python
def filter_patch_list(self, patch_list):
    return get_patches_with_missing_features(filesystem, patches_folder, patch_list, features)
```

where the `features` parameter defines the list of features which must be present if the patch is to be skipped.
