"""Implements the command line interface for `eo-grow`."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import Any

import click
import ray

from .core.config import RawConfig, collect_configs_from_path, interpret_config_from_dict
from .core.logging import CLUSTER_FILE_LOCATION_ON_HEAD
from .core.schemas import build_schema_template
from .pipelines.testing import TestPipeline
from .utils.general import jsonify
from .utils.meta import collect_schema, import_object, load_pipeline_class
from .utils.ray import generate_cluster_config_path, start_cluster_if_needed

variables_option = click.option(
    "-v",
    "--variable",
    "cli_variables",
    multiple=True,
    type=str,
    help='Specify variables to use in config (overriding present config values). Must be of form `"name:value"`',
)
test_patches_option = click.option(
    "-t",
    "--test",
    "test_patches",
    multiple=True,
    type=int,
    help=(
        "One or more indices of EOPatches. If given, a pipeline will run only for those EOPatches. "
        "Example: -t 0 -t 42 will run the pipeline for EOPatches with indices 0 and 42."
    ),
)
ray_remote_kwargs_option = click.option(
    "-r",
    "--ray_remote_kwargs",
    "ray_remote_kwargs",
    multiple=True,
    type=str,
    help=(
        'Specify parameters passed to `ray.remote` in form of `"num_cpus:2"` or `"resources.my_resource:1"` for values'
        " passed to the `resources` parameter of `ray.remote`."
    ),
)


@click.command()
@click.argument("config_path", type=click.Path())
@variables_option
@test_patches_option
@ray_remote_kwargs_option
def run_pipeline(
    config_path: str, cli_variables: tuple[str, ...], test_patches: tuple[int, ...], ray_remote_kwargs: tuple[str, ...]
) -> None:
    """Execute eo-grow pipeline using CLI.

    \b
    Example:
        eogrow config_files/config.json
    """

    raw_configs = collect_configs_from_path(config_path)
    cli_variable_mapping = dict(_parse_cli_mapping(cli_var) for cli_var in cli_variables)
    ray_kwargs = _parse_ray_remote_kwargs(ray_remote_kwargs)

    configs = []
    for raw_config in raw_configs:
        config = interpret_config_from_dict(raw_config, cli_variable_mapping)
        if test_patches:
            config["test_subset"] = list(test_patches)

        load_pipeline_class(config).from_raw_config(config)  # quickly validates all pipelines
        configs.append(config)

    for config in configs:
        if config.get("debug", False):
            load_pipeline_class(config).from_raw_config(config).run()
        else:
            ray.init(address="auto", ignore_reinit_error=True)
            ray.get(_pipeline_spawner.options(**ray_kwargs).remote(config))  # type: ignore[attr-defined]


@ray.remote
def _pipeline_spawner(config: RawConfig) -> None:
    load_pipeline_class(config).from_raw_config(config).run()


@click.command()
@click.argument("cluster_yaml", type=click.Path())
@click.argument("config_path", type=click.Path())
@click.option(
    "--start", "start_cluster", is_flag=True, type=bool, help="Starts the cluster if it is not currently running."
)
@click.option(
    "--tmux",
    "use_tmux",
    is_flag=True,
    type=bool,
    help="Run the cluster in a detached mode using tmux software. Use Ctrl+B and d to detach.",
)
@variables_option
@test_patches_option
@ray_remote_kwargs_option
def run_pipeline_on_cluster(
    config_path: str,
    cluster_yaml: str,
    start_cluster: bool,
    use_tmux: bool,
    cli_variables: tuple[str, ...],
    test_patches: tuple[int, ...],
    ray_remote_kwargs: str,
) -> None:
    """Command for running an eo-grow pipeline on a remote Ray cluster of AWS EC2 instances. The provided config is
    fully constructed and uploaded to the cluster head in the `~/.synced_configs/` directory, where it is then
    executed. A custom suffix is added to distinguish runs which use the same config multiple times.

    \b
    Example:
        eogrow-ray cluster.yaml config_files/config.json
    """
    if start_cluster:
        start_cluster_if_needed(cluster_yaml)

    raw_configs = [interpret_config_from_dict(config) for config in collect_configs_from_path(config_path)]
    remote_path = generate_cluster_config_path(config_path)

    with NamedTemporaryFile(mode="w", delete=True, suffix=".json") as local_path:
        json.dump(raw_configs, local_path)
        local_path.flush()  # without this the sync can happen before the file content is written

        subprocess.run(f"ray rsync_up {cluster_yaml} {local_path.name!r} {remote_path!r}", shell=True)
        subprocess.run(f"ray rsync_up {cluster_yaml} {cluster_yaml!r} {CLUSTER_FILE_LOCATION_ON_HEAD!r}", shell=True)

    cmd = (
        f"eogrow {remote_path}"
        + "".join(f' -v "{cli_var_spec}"' for cli_var_spec in cli_variables)  # B028
        + "".join(f" -t {patch_index}" for patch_index in test_patches)
        + "".join(f' -r "{ray_kwarg}"' for ray_kwarg in ray_remote_kwargs)  # B028
    )
    exec_flags = "--tmux" if use_tmux else ""

    subprocess.run(f"ray exec {exec_flags} {cluster_yaml} {cmd!r}", shell=True)  # B028


@click.command()
@click.argument("import_path", type=str)
@click.argument("template_path", type=click.Path(), required=False)
@click.option(
    "-f",
    "--force",
    "force_override",
    is_flag=True,
    type=bool,
    help=(
        "In case a template path is provided and a file in the path already exists this flag is used to force "
        "override it."
    ),
)
@click.option(
    "--template-format",
    "template_format",
    type=click.Choice(["minimal", "open-api"], case_sensitive=False),
    help="Specifies which template format to use. The default is `minimal`",
    default="minimal",
)
@click.option(
    "--required-only",
    "required_only",
    is_flag=True,
    type=bool,
    help="If provided it will only include required fields in the template. Only for `minimal` template format",
)
@click.option(
    "--add-descriptions",
    "add_descriptions",
    is_flag=True,
    type=bool,
    help="Adds descriptions to template. Only for `minimal` template format",
)
def make_template(
    import_path: str,
    template_path: str | None,
    force_override: bool,
    template_format: str,
    required_only: bool,
    add_descriptions: bool,
) -> None:
    """Command for creating a config template for an eo-grow pipeline

    \b
    Examples:
        - save template to file:
            eogrow-template eogrow.pipelines.download.DownloadPipeline config_files/download.json
        - print template to command line:
            eogrow-template eogrow.pipelines.download.DownloadPipeline
    """
    if not force_override and template_path and os.path.isfile(template_path):
        raise FileExistsError(f"File {template_path} already exists. You can use -f to force override it.")

    class_with_schema = import_object(import_path)
    schema = collect_schema(class_with_schema)

    if template_format == "open-api":
        template = schema.schema()
    else:
        template = build_schema_template(
            schema,
            pipeline_import_path=import_path,
            required_only=required_only,
            add_descriptions=add_descriptions,
        )

    if template_path:
        with open(template_path, "w") as file:
            json.dump(template, file, indent=2, default=jsonify)
    else:
        click.echo(json.dumps(template, indent=2, default=jsonify))


@click.command()
@click.argument("config_path", type=click.Path())
def validate_config(config_path: str) -> None:
    """Validate config without running a pipeline.

    \b
    Example:
        eogrow-validate config_files/config.json
    """
    for config in collect_configs_from_path(config_path):
        raw_config = interpret_config_from_dict(config)
        load_pipeline_class(config).Schema.parse_obj(raw_config)

    click.echo("Config validation succeeded!")


@click.command()
@click.argument("config_path", type=click.Path())
def run_test_pipeline(config_path: str) -> None:
    """Runs a test pipeline that only makes sure the managers work correctly. This can be used to select best
    area manager parameters.

    \b
    Example:
        eogrow-test any_pipeline_config.json
    """
    for crude_config in collect_configs_from_path(config_path):
        raw_config = interpret_config_from_dict(crude_config)
        pipeline = TestPipeline.with_defaults(raw_config)
        pipeline.run()


def _parse_cli_mapping(mapping_str: str) -> tuple[str, str]:
    """Checks that the input is of shape `name:value` and then splits it into a tuple"""
    match = re.match(r"(?P<name>.+?):(?P<value>.+)", mapping_str)
    if match is None:
        raise ValueError(f'CLI input {mapping_str} is not of form `"name:value"`')
    parsed = match.groupdict()
    return parsed["name"], parsed["value"]


def _parse_ray_remote_kwargs(ray_remote_kwargs: tuple[str, ...]) -> dict[str, Any]:
    ray_kwargs: dict[str, Any] = defaultdict(dict)
    for param, value in map(_parse_cli_mapping, ray_remote_kwargs):
        kwargs = ray_kwargs["resources"] if param.startswith("resources.") else ray_kwargs
        kwargs[param] = float(value)
    return ray_kwargs
