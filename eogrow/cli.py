"""
Module implementing command line interface for `eo-grow`
"""
import json
import os
import re
import subprocess
from typing import Optional, Tuple

import click

from .core.config import collect_configs_from_path, decode_config_list, encode_config_list, interpret_config_from_dict
from .core.schemas import build_minimal_template, build_schema_template
from .pipelines.testing import TestPipeline
from .utils.general import jsonify
from .utils.meta import collect_schema, import_object, load_pipeline_class

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


class EOGrowCli:
    """A command line interface class for `eo-grow`

    It is designed to be transferable to other packages based on `eo-grow`.

    Note: It looks that click doesn't allow implementing CLI in a proper class structure, therefore here we are using
    class variable to store the list of packages (https://github.com/pallets/click/issues/601).
    """

    _command_namespace = "eogrow"

    def __init__(self, command_namespace: Optional[str] = None):
        """
        :param command_namespace: A namespace for calling CLI, e.g. command_namespace='eogrow' if you call commands
            "eogrow config.json".
        """
        if command_namespace:
            EOGrowCli._command_namespace = command_namespace

    @staticmethod
    @click.command()
    @click.argument("config_filename_or_string", type=click.Path())
    @variables_option
    @test_patches_option
    @click.option(
        "-e",
        "--encoding",
        "encoding",
        is_flag=True,
        type=bool,
        help="The string passed to method is treated as an encoded config instead of filename.",
    )
    def main(
        config_filename_or_string: str, cli_variables: Tuple[str], test_patches: Tuple[int], encoding: bool
    ) -> None:
        """Execute eo-grow pipeline using CLI.

        \b
        Example:
            eogrow config_files/config.json
        """
        if encoding:
            raw_configs = decode_config_list(config_filename_or_string)
        else:
            raw_configs = collect_configs_from_path(config_filename_or_string)

        cli_variable_mapping = dict(_parse_cli_variable(cli_var) for cli_var in cli_variables)
        pipelines = []
        for raw_config in raw_configs:
            config = interpret_config_from_dict(raw_config, cli_variable_mapping)
            if test_patches:
                config["patch_list"] = list(test_patches)

            pipelines.append(load_pipeline_class(config).from_raw_config(config))

        for pipeline in pipelines:
            pipeline.run()

    @staticmethod
    @click.command()
    @click.argument("cluster_yaml", type=click.Path())
    @click.argument("config_filename", type=click.Path())
    @click.option(
        "--start", "start_cluster", is_flag=True, type=bool, help="Starts the cluster if it is not currently running."
    )
    @click.option(
        "--stop",
        "stop_cluster",
        is_flag=True,
        type=bool,
        help=(
            "Stops the cluster if after running the pipeline. In order for this to work got to AWS console "
            "-> IAM -> Roles -> select ray-autoscaler-v1 role and attach IAMReadOnlyAccess policy."
        ),
    )
    @click.option(
        "--screen",
        "use_screen",
        is_flag=True,
        type=bool,
        help=(
            "Run the cluster in a detached mode using screen software. Use Ctrl+A+D to detach any time, "
            "even when running a pipeline or a Jupyter notebook. Use Ctrl+D to terminate the remote screen."
        ),
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
    def ray(
        config_filename: str,
        cluster_yaml: str,
        start_cluster: bool,
        stop_cluster: bool,
        use_screen: bool,
        use_tmux: bool,
        cli_variables: Tuple[str],
        test_patches: Tuple[int],
    ) -> None:
        """Command for running an eo-grow pipeline on a remote Ray cluster of AWS EC2 instances

        \b
        Example:
            eogrow-ray cluster.yaml config_files/config.json
        """
        if stop_cluster and (use_screen or use_tmux):
            raise NotImplementedError("It is not clear how to combine stop flag with either screen or tmux flag")

        configs = collect_configs_from_path(config_filename)
        encoded_configs = encode_config_list(configs)
        cmd = (
            f"{EOGrowCli._command_namespace} -e {encoded_configs}"
            + "".join(f' -v "{cli_var_spec}"' for cli_var_spec in cli_variables)
            + "".join(f" -t {patch_index}" for patch_index in test_patches)
            + ("; " if stop_cluster else "")  # Otherwise, ray will incorrectly prepare a command for stopping a cluster
        )
        flag_info = [("start", start_cluster), ("stop", stop_cluster), ("screen", use_screen), ("tmux", use_tmux)]
        exec_flags = " ".join(f"--{flag_name}" for flag_name, use_flag in flag_info if use_flag)

        subprocess.call(f"ray exec {exec_flags} {cluster_yaml} '{cmd}'", shell=True)

    @staticmethod
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
        type=click.Choice(["minimal", "open-api", "full"], case_sensitive=False),
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
        template_path: Optional[str],
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

        object_with_schema = import_object(import_path)
        schema = collect_schema(object_with_schema)

        if template_format == "open-api":
            template = schema.schema()
        elif template_format == "full":
            template = build_schema_template(schema)
        else:
            template = build_minimal_template(
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

    @staticmethod
    @click.command()
    @click.argument("config_filename", type=click.Path())
    def validate_config(config_filename: str) -> None:
        """Validate config without running a pipeline.

        \b
        Example:
            eogrow-validate config_files/config.json
        """
        for config in collect_configs_from_path(config_filename):
            raw_config = interpret_config_from_dict(config)
            load_pipeline_class(config).Schema.parse_obj(raw_config)

        click.echo("Config validation succeeded!")

    @staticmethod
    @click.command()
    @click.argument("config_filename", type=click.Path())
    def run_test_pipeline(config_filename: str) -> None:
        """Runs a test pipeline that only makes sure the managers work correctly. This can be used to select best
        area manager parameters.

        \b
        Example:
            eogrow-test any_pipeline_config.json
        """
        for crude_config in collect_configs_from_path(config_filename):
            raw_config = interpret_config_from_dict(crude_config)
            pipeline = TestPipeline.with_defaults(raw_config)
            pipeline.run()


def _parse_cli_variable(mapping_str: str) -> Tuple[str, str]:
    """Checks that the input is of shape `name:value` and then splits it into a tuple"""
    match = re.match(r"(?P<name>.+?):(?P<value>.+)", mapping_str)
    if match is None:
        raise ValueError(f'CLI variable input {mapping_str} is not of form `"name:value"`')
    parsed = match.groupdict()
    return parsed["name"], parsed["value"]
