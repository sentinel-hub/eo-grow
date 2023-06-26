
## [Version 1.5.1] - 2023-05-03

- Fix bug in `LoggingManager.Schema` where `Tuple[str]` was used instead of `Tuple[str, ...]` for certain fields, preventing parsing of correct configurations.


## [Version 1.5.0] - 2023-04-25

- (**code-breaking**) Simplified `RasterizePipeline` and improve rasterization of temporal vectors.
- (**code-breaking**) Area managers no longer offer AOI modification in the `area` parameter. It has been replaced with a simpler `filename` field. We added a rerouting parser, so old configs should work for a while longer.
- (**code-breaking**) Separated machine learning requirements to `ML` extra that you can install via `pip install eogrow[ML]`. These packages are only necessary for sampling, training, and prediction pipelines.
- Added `VectorImportPipeline` for adding vector features to EOPatches.
- Improved `ExportMapsPipeline` when working with large amounts of files, contributed by @aashishd.
- Config files are now uploaded to the cluster before being executed. This prevents issues with commands failing on very large configs.
- Added `restrict_types` validator that detects incompatible `FeatureType` inputs for fields of type `Feature`.
- Added `ensure_storage_key_presence` validator, which checks that the specified storage key is defined in the storage manager. Typos in storage keys will now be detected at validation.
- Storage managers now support a `filesystem_kwargs` parameter.
- Fixed bug where area managers would not filter the grid correctly if the grid was read from the cache.
- Logs to stdout are now colored and contain timestamps.
- Logging configs can now use `"..."` value to reference default packages for fields such as `pipeline_ignore_packages`.
- Pipelines can now be given custom names, which makes it easier to identify certain pipelines when searching for logs or when running them in config chains.
- Switched to a `pyproject.toml` based installation.
- Added new sections to documentation of the high level overview and a collection of commonly used patterns.
- Improved testing tools.
- Various minor improvements.


## [Version 1.4.0] - 2023-01-26

- (**code-breaking**) Large changes to area managers. See PR https://github.com/sentinel-hub/eo-grow/pull/168
    * EOPatch manager functionality was merged to area managers. EOPatch managers were removed.
    * Changes to area manager Schemas.
    * Changes to area manager interface. Check documentations for all the changes.
    * Adjustments to Pipeline interface. See PR https://github.com/sentinel-hub/eo-grow/pull/168 for how most pipelines need to be adjusted.
    * Improved filtration via list of EOPatch names.
- (**code-breaking**) Added `ZipMapPipeline` which replaces `MappingPipeline`.
- (**code-breaking**) Added `SplitGridPipeline` which replaces `SwitchGridsPipeline`.
- (**code-breaking**) Adjusted resize parameters in `ImportTiffPipeline` according to changes in `SpatialResizeTask` in new `eo-learn` version.
- Fixed issue with label encoder in prediction pipeline. Contributed by @ashishdhiman-tomtom
- Moved types to `eogrow.types` and deprecate `eogrow.utils.types`. Remove `Path` type alias.
- Added support for EOPatch names when using the `-t` flag.


## [Version 1.3.3] - 2022-17-11

- Added `ImportTiffPipeline` for importing a tiff file into EOPatches.
- `ExportMapsPipeline` now runs in parallel (single-machine only).
- Fixed issue where `ExportMapsPipeline` consumed increasing amounts of storage space.
- Area and eopatch managers for batch grids now warn the user if not linked correctly.
- Added `pyogrio` as a possible `geopandas` backend for IO (experimental).
- Add support for `geopandas` version 0.12.
- Improve types after `mypy` version 0.990.
- Removed `utils.enum` and old style of templating due to non-use.
- Other various improvements and clean-ups.


## [Version 1.3.2] - 2022-24-10

- Greatly improved `ExportMapsPipeline` and `IngestByocTilesPipeline`, which are now also able to export and ingest temporal BYOC collections
- Improved test suite for exporting maps and ingesting BYOC collections
- Fixed code according to newly exposed `eolearn.core` types
- Fixed broken github links in documentation
- Improvements to CI, added pre-commit hooks to the repository


## [Version 1.3.1] - 2022-31-08

- BYOC ingestion pipeline is better at handling CRS objects
- Becaue `pydantic` now type-checks default factories two custom factories `list_factory` and `dict_factory` have been added, because using just `list` currently clashes with fields of kind `List[int]`.


## [Version 1.3.0] - 2022-30-08

- Added `IngestByocTiles` pipeline, which creates or updates a BYOC collection from maps exported via `ExportMapsPipeline`.
- Greatly improved `DataCollection` parser, which can now parse `DataCollectionSchema` objects instead of just names.
- Added tests for validator utility functions.
- New general validators `ensure_defined_together` and `ensure_exactly_one_defined` for verifying optional parameters.
- Documentation of `Schema` objects is now much more verbose.
- `ExportMapsPipeline` now saves maps into subfolders (per UTM zone).
- Fixed issue where `ExportMapPipeline` ignored `dtype` and `nodata` when merging.
- Improved handling of `aws_profile` parameter in storage managers.
- `RasterizePipeline` now has an additional `raster_shape` parameter.


## [Version 1.2.0] - 2022-27-07

- Fixed a bug in `BatchToEOPatchPipeline` where temporal dimension of some imported features could be reversed. Memory-optimization functionalities have been reverted.
- Improved the way `filesystem` object is passed to EOTasks in EOWorkflows. These changes are a consequence of changes in `eo-learn==1.2.0`.
- Added support for `aws_acl` parameter into `Storage` schema.
- Download pipelines now support an optional `size` parameter.
- Official support for Python `3.10`.
- Large changes in testing utilities. Statistics produced by `ContentTester` have been changed and are now more descriptive.
- Improvements in code-style checkers and CI.


## [Version 1.1.1] - 2022-14-06

- Support session sharing in download pipelines.
- Improved `BatchAreaManager` bounding boxes.
- Improve memory footprint of various pipelines.
- Disabled `skip_existing` and `eopatch_list` at validation time for pipelines that do not support filtration.
- Support for rasterization of temporal vector features from files.
- Docs are now built automatically and the type annotations are included in parameter descriptions, resulting in better readability.
- Many minor improvements and fixes in code, tests, and documentation.


## [Version 1.1.0] - 2022-03-05

- Large changes in config objects and schemas:
  * replaced `Config` object with config utility functions `collect_configs_from_path`, `interpret_config_from_dict`, and `interpret_config_from_path`,
  * pipeline and manager config objects are now `pydantic` schema classes, which are fully typed objects,
  * removed `${env:variable}` from the config language.

- Changes in area managers:
  * added `AreaManager.cache_grid` method,
  * (**code-breaking**)improved functionalities of `BatchAreaManger`, instead of `tile_buffer` it now uses `tile_buffer_x` and `tile_buffer_y` config parameters,
  * (**code-breaking**) improved `UtmZoneAreaManager`, replaced `patch_buffer` config parameter with `patch_buffer_x` and `patch_buffer_y` which now work with absolute instead of relative buffers ,
  * implemented grid transformation methods for `UtmZoneAreaManager` and `BatchAreaManager`.

- Other core improvements:
  * added `EOGrowObject.from_raw_config` and `EOGrowObject.from_path` methods,
  * fixed an issue in `EOPatchManager`,
  * improvements of pipeline logging, logging handlers, and filters.

- Pipeline improvements:
  * Implemented `SwitchGridPipeline` for converting data between tiling grids.
  * Large updates of `BatchDownloadPipeline` with restructured config schema and additional functionalities.
  * `BatchToEOPatchPipeline` now works with `input_folder_key` and `output_folder_key` instead of `folder_key` and has an option not to delete input data. A few issues in the pipeline were fixed and unit tests were added.
  * Minor improvements of config parameters in `MergeSamplesPipeline` and prediction pipelines.
  * Implemented `DummyDataPipeline` for generating data for unit tests.
- New tasks:
  * `SpatialJoinTask` and `SpatialSliceTask` for spatial operations on EOPatches,
  * `DummyRasterFeatureTask` and `DummyTimestampFeatureTask` for creating EOPatches with dummy data.
- Updates in utilities:
  * added utilities for spatial operations and grid transformations,
  * implemented `eogrow.utils.fs.LocalFolder` abstraction,
  * renamed `get_patches_without_all_features` into `get_patches_with_missing_features` from `eogrow.utils.filter` ,
  * (**code-breaking**) updated `eogrow.utils.testing.run_and_test_pipeline` to work with a list of pipeline configs.
- Created the `eo-grow` package [documentation page](https://eo-grow.readthedocs.io/en/latest/).
- `eo-grow` is now a fully typed package. Added mypy and isort code checking to CI.
- Updated tutorial notebooks to work with the latest code.
- Many minor improvements and fixes in code, tests, and documentation.


## [Version 1.0.0] - 2022-02-10

First release of the `eo-grow` package.