"""Implements the command line interface for `eo-grow`."""

from __future__ import annotations

import json
import os
import re
import subprocess
from tempfile import NamedTemporaryFile
from typing import Iterable

import click

from .core.config import CrudeConfig, RawConfig, collect_configs_from_path, interpret_config_from_dict
from .core.logging import CLUSTER_FILE_LOCATION_ON_HEAD
from .core.schemas import build_schema_template
from .utils.general import jsonify
from .utils.meta import collect_schema, import_object, load_pipeline_class
from .utils.pipeline_chain import run_pipeline_chain, validate_pipeline_chain
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


@click.command()
@click.argument("config_path", type=click.Path())
@variables_option
@test_patches_option
def run_pipeline(config_path: str, cli_variables: tuple[str, ...], test_patches: tuple[int, ...]) -> None:
    """Execute eo-grow pipeline using CLI.

    \b
    Example:
        eogrow config_files/config.json
    """

    crude_config = collect_configs_from_path(config_path)
    cli_variable_mapping = dict(_parse_cli_variable(cli_var) for cli_var in cli_variables)

    if isinstance(crude_config, dict):
        config = _prepare_config(crude_config, cli_variable_mapping, test_patches)
        pipeline = load_pipeline_class(config).from_raw_config(config)
        pipeline.run()

    else:
        pipeline_chain = [_prepare_config(config, cli_variable_mapping, test_patches) for config in crude_config]
        validate_pipeline_chain(pipeline_chain)
        run_pipeline_chain(pipeline_chain)


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
def run_pipeline_on_cluster(
    config_path: str,
    cluster_yaml: str,
    start_cluster: bool,
    use_tmux: bool,
    cli_variables: tuple[str, ...],
    test_patches: tuple[int, ...],
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

    remote_path = generate_cluster_config_path(config_path)
    with NamedTemporaryFile(mode="w", delete=True, suffix=".json") as local_path:
        json.dump(collect_configs_from_path(config_path), local_path)
        local_path.flush()  # without this the sync can happen before the file content is written

        subprocess.run(f"ray rsync_up {cluster_yaml} {local_path.name!r} {remote_path!r}", shell=True)
        subprocess.run(f"ray rsync_up {cluster_yaml} {cluster_yaml!r} {CLUSTER_FILE_LOCATION_ON_HEAD!r}", shell=True)

    cmd = (
        f"eogrow {remote_path}"
        + "".join(f' -v "{cli_var_spec}"' for cli_var_spec in cli_variables)  # B028
        + "".join(f" -t {patch_index}" for patch_index in test_patches)
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
    config = collect_configs_from_path(config_path)
    if isinstance(config, dict):
        pipeline_config = _prepare_config(config, {}, ())
        collect_schema(load_pipeline_class(pipeline_config)).parse_obj(pipeline_config)
    else:
        for i, run_config in enumerate(config):
            if "pipeline_config" not in run_config:
                raise ValueError(f"Pipeline-chain element {i} is missing the field `pipeline_config`.")
            run_config["pipeline_config"] = _prepare_config(run_config["pipeline_config"], {}, ())
        validate_pipeline_chain(config)  # type: ignore[arg-type]

    click.echo("Config validation succeeded!")


def _prepare_config(config: CrudeConfig, variables: dict[str, str], test_patches: Iterable[int]) -> RawConfig:
    raw_config = interpret_config_from_dict(config, variables)
    if test_patches:
        raw_config["test_subset"] = list(test_patches)
    return raw_config


def _parse_cli_variable(mapping_str: str) -> tuple[str, str]:
    """Checks that the input is of shape `name:value` and then splits it into a tuple"""
    match = re.match(r"(?P<name>.+?):(?P<value>.+)", mapping_str)
    if match is None:
        raise ValueError(f'CLI variable input {mapping_str} is not of form `"name:value"`')
    parsed = match.groupdict()
    return parsed["name"], parsed["value"]
