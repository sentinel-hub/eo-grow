"""Implementation of LoggingManager and different handlers used for logging."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from logging import FileHandler, Filter, Formatter, Handler, LogRecord
from typing import Any, Sequence, Tuple

import colorlog
import fs
import fs.copy
from fs.base import FS
from fs.errors import FilesystemClosed
from fs.osfs import OSFS
from pydantic import Field

from eolearn.core.utils.fs import join_path, unpickle_fs

from ..utils.eopatch_list import save_names
from ..utils.fs import LocalFile
from ..utils.general import jsonify
from ..utils.logging import get_instance_info
from ..utils.meta import get_package_versions
from ..utils.ray import CLUSTER_CONFIG_DIR
from .base import EOGrowObject
from .config import RawConfig
from .schemas import ManagerSchema
from .storage import StorageManager

DEFAULT_PACKAGES_TOKEN = "..."  # fix docs if this is changed!
CLUSTER_FILE_LOCATION_ON_HEAD = fs.path.join(CLUSTER_CONFIG_DIR, "cluster.yaml")


class LoggingManager(EOGrowObject):
    """A class that manages logging specifics"""

    class Schema(ManagerSchema):
        save_logs: bool = Field(
            False,
            description=(
                "A flag to determine if pipeline logs and reports will be saved to "
                "logs folder. This includes potential EOExecution reports and logs."
            ),
        )
        include_logs_to_report: bool = Field(
            False,
            description=(
                "If log files should be parsed into an EOExecution report file or just linked. When working "
                "with larger number of EOPatches the recommended option is False."
            ),
        )
        eoexecution_ignore_packages: Tuple[str, ...] = Field(
            (DEFAULT_PACKAGES_TOKEN,),
            description=(
                "Names of packages for which the logs will not be written to EOExecution log files. You can reference"
                ' the defaults with "...", for example, adding another package can be done with ["...", "some_package"]'
            ),
        )

        pipeline_ignore_packages: Tuple[str, ...] = Field(
            (DEFAULT_PACKAGES_TOKEN,),
            description=(
                "Names of packages for which the logs will not be written to the main pipeline log file. You can"
                ' reference the defaults with "...", for example, adding another package can be done with'
                ' ["...", "some_package"]'
            ),
        )
        pipeline_logs_backup_interval: float = Field(
            60,
            description=(
                "When working with a remote storage this parameter defines a minimal number of seconds between "
                "two consecutive times when pipeline log file will be copied into the remote storage."
            ),
        )

        show_logs: bool = Field(False, description="Shows basic pipeline execution logs at stdout.")
        stdout_log_packages: Tuple[str, ...] = Field(
            (DEFAULT_PACKAGES_TOKEN,),
            description=(
                "Names of packages for which the logs will be written to stdout. You can reference the defaults with"
                ' "...", for example, adding another package can be done with ["...", "package_to_display"]'
            ),
        )

        capture_warnings: bool = Field(
            True,
            description=(
                "If warnings should be treated as logs and with save_logs=True written into log files instead of "
                "being printed in stderr."
            ),
        )

    config: Schema

    def __init__(self, config: Schema, storage: StorageManager):
        """
        :param config: A configuration file
        :param storage: An instance of StorageManager class
        """
        super().__init__(config)

        self.storage = storage

    def get_pipeline_logs_folder(self, pipeline_execution_name: str, full_path: bool = False) -> str:
        """Provides path to the folder where logs of this pipeline execution will be saved

        :param pipeline_execution_name: Name of current pipeline execution
        :param full_path: If it should provide a full absolute path or a path relative to the filesystem object
        """
        main_logs_folder = self.storage.get_logs_folder(full_path=full_path)
        if full_path:
            return join_path(main_logs_folder, pipeline_execution_name)
        return fs.path.combine(main_logs_folder, pipeline_execution_name)

    def start_logging(self, logger: logging.Logger, pipeline_execution_name: str, filename: str) -> list[Handler]:
        """Creates a folder for logs and sets up (and returns) logging handlers

        Supported handlers:
        - Writing to a file in pipeline logs folder
        - Printing logs to a standard output
        """
        if self.config.save_logs:
            logs_folder = self.get_pipeline_logs_folder(pipeline_execution_name)
            self.storage.filesystem.makedirs(logs_folder, recreate=True)
            self._add_cluster_config_to_logs(logs_folder)

        logger.setLevel(logging.DEBUG)

        for default_handler in logger.handlers:
            default_handler.setLevel(logging.WARNING)

        new_handlers: list[Handler] = []

        if self.config.save_logs:
            file_handler = self._create_file_handler(pipeline_execution_name, filename)
            logger.addHandler(file_handler)
            new_handlers.append(file_handler)

        if self.config.show_logs:
            stdout_handler = self._create_stdout_handler()
            logger.addHandler(stdout_handler)
            new_handlers.append(stdout_handler)

        if self.config.capture_warnings:
            logging.captureWarnings(True)

        return new_handlers

    def _add_cluster_config_to_logs(self, logs_folder: str) -> None:
        """If it detects a synced `cluster.yaml` file, it will copy it to the logs folder."""
        os_path = CLUSTER_FILE_LOCATION_ON_HEAD.replace("~", os.path.expanduser("~"))
        if os.path.exists(os_path):
            os_folder, os_file = fs.path.split(CLUSTER_FILE_LOCATION_ON_HEAD)
            os_fs = OSFS(os_folder)  # the file is on the head node, might not be visible in storage.filesystem
            fs.copy.copy_file(os_fs, os_file, self.storage.filesystem, fs.path.join(logs_folder, "cluster.yaml"))

    def _create_file_handler(self, pipeline_execution_name: str, filename: str) -> Handler:
        """Creates a logging handler to write a pipeline log to a file."""
        logs_filename = fs.path.combine(self.get_pipeline_logs_folder(pipeline_execution_name), filename)
        file_handler = RegularBackupHandler(
            logs_filename,
            filesystem=self.storage.filesystem,
            backup_interval=self.config.pipeline_logs_backup_interval,
            encoding="utf-8",
        )

        formatter = Formatter(
            "%(levelname)s [%(asctime)s] %(name)s:%(lineno)d:\n\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        file_handler.addFilter(LogFileFilter(ignore_packages=self.config.pipeline_ignore_packages))

        return file_handler

    def _create_stdout_handler(self) -> Handler:
        """Creates a logging handler to write logs into a standard output."""
        stdout_handler = colorlog.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s %(levelname)s %(cyan)s[%(asctime)s] %(light_blue)s%(name)s:%(lineno)d: %(white)s%(message)s",
            datefmt="%H:%M:%S",
        )
        stdout_handler.setFormatter(formatter)

        stdout_handler.addFilter(StdoutFilter(log_packages=self.config.stdout_log_packages))

        return stdout_handler

    def stop_logging(self, logger: logging.Logger, handlers: list[Handler]) -> None:
        """Updates logs, removes pipeline handlers from the global logger and puts global logging level back to
        default
        """
        if self.config.capture_warnings:
            logging.captureWarnings(False)

        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

        logger.setLevel(logging.WARNING)

    def update_pipeline_report(
        self,
        pipeline_execution_name: str,
        pipeline_config: EOGrowObject.Schema,
        pipeline_raw_config: RawConfig | None,
        pipeline_id: str,
        pipeline_timestamp: str,
        elapsed_time: float | None = None,
    ) -> None:
        """A method in charge of preparing a report about pipeline run.

        Content of a report:
         - pipeline configuration parameters,
         - pipeline execution stats,
         - versions of Python and Python packages,
         - information about a compute instance on which the pipeline is running.
        """
        if not self.config.save_logs:
            return

        report = {
            "config_parameters": pipeline_raw_config,
            "execution_parameters": repr(pipeline_config),
            "pipeline_execution_stats": {
                "pipeline_id": pipeline_id,
                "start_time": pipeline_timestamp,
                "elapsed_time": "<Not yet finished>" if elapsed_time is None else elapsed_time,
            },
            "versions": {"Python": sys.version, **get_package_versions()},
            "instance_info": get_instance_info(),
        }

        report_filename = fs.path.combine(
            self.get_pipeline_logs_folder(pipeline_execution_name), "pipeline-report.json"
        )
        with self.storage.filesystem.open(report_filename, "w") as report_file:
            json.dump(report, report_file, indent=2, default=jsonify)

    def save_eopatch_execution_status(
        self, pipeline_execution_name: str, finished: list[str], failed: list[str]
    ) -> None:
        """Saves lists of EOPatch names for which execution either finished successfully or failed"""
        if not self.config.save_logs:
            return

        logs_folder = self.get_pipeline_logs_folder(pipeline_execution_name)
        for eopatches, filename in [(finished, "finished.json"), (failed, "failed.json")]:
            save_names(self.storage.filesystem, fs.path.combine(logs_folder, filename), eopatches)


class FilesystemHandler(FileHandler):
    """A filesystem abstraction of FileHandler

    In case the handler gets a local path it behaves the same as FileHandler. In case it gets a remote path it writes
    logs first to a local path and then copies them to the remote location.

    IMPORTANT: This handler will by default have an extra `FilesystemFilter` which will ignore logs from packages that
    produce logs during `LocalFile.copy_to_remote` call. Otherwise, a log that would be created within an `emit`
    call would be recursively sent back to the handler. That would either trigger an infinite recursion or make the
    process stuck waiting for a thread lock release.
    """

    def __init__(self, path: str, filesystem: FS | bytes, encoding: str | None = "utf-8", **kwargs: Any):
        """
        :param path: A path to a log file that is relative to the given `filesystem` object.
        :param filesystem: A filesystem to where logs will be written. It can either be an instance of a filesystem
            object or its pickled copy.
        :param encoding: Encoding used to write log files.
        :param kwargs: Keyword arguments that will be propagated to FileHandler.
        """
        filesystem_object = unpickle_fs(filesystem) if isinstance(filesystem, bytes) else filesystem
        self.local_file = LocalFile(path, mode="w", filesystem=filesystem_object)

        super().__init__(self.local_file.path, encoding=encoding, **kwargs)

        self.addFilter(FilesystemFilter())

    def close(self) -> None:
        """Closes logging and closes the local file"""
        super().close()
        with contextlib.suppress(FilesystemClosed):
            self.local_file.close()


class RegularBackupHandler(FilesystemHandler):
    """A customized FilesystemHandler that makes a copy to a remote location regularly after given amount of time."""

    def __init__(self, path: str, filesystem: FS | bytes, backup_interval: float | int, **kwargs: Any):
        """
        :param path: A path to a log file that is relative to the given `filesystem` object.
        :param filesystem: A filesystem to where logs will be written. It can either be an instance of a filesystem
            object or its pickled copy.
        :param backup_interval: A minimal number of seconds before handler will back up the log file to the remote
            location. The backup will only happen when the next log record will be emitted.
        """
        super().__init__(path=path, filesystem=filesystem, **kwargs)

        self.backup_interval = backup_interval
        self._last_backup_time = time.monotonic()

    def emit(self, record: LogRecord) -> None:
        """Save a new record and backup to remote if the backup hasn't been done in the given amount of time."""
        super().emit(record)

        if time.monotonic() > self._last_backup_time + self.backup_interval:
            self.local_file.copy_to_remote()
            self._last_backup_time = time.monotonic()


class EOExecutionHandler(FilesystemHandler):
    """A customized FilesystemHandler that makes a copy to a remote location every time a new node in a workflow
    is started.
    """

    def emit(self, record: LogRecord) -> None:
        """Save a new record. In case a new node in EOWorkflow is started it will copy the log file to remote."""
        super().emit(record)

        if record.name == "eolearn.core.eoworkflow" and record.message.startswith("Computing"):
            self.local_file.copy_to_remote()


class FilesystemFilter(Filter):
    """The sole purpose of this filter is to capture any log that happens during `LocalFile.copy_to_remote` call. Any
    log that would not be captured would break the entire runtime.
    """

    IGNORE_HARMFUL_LOGS = (
        "botocore",
        "boto3.resources",
        "s3transfer",
    )

    def filter(self, record: LogRecord) -> bool:
        """Ignores logs from certain low-level packages"""
        return not record.name.startswith(self.IGNORE_HARMFUL_LOGS)


class StdoutFilter(Filter):
    """Filters log messages passed to standard output"""

    DEFAULT_LOG_PACKAGES = (
        "eogrow",
        "__main__",
        "root",
        "sentinelhub.api.batch",
    )

    def __init__(self, *args: Any, log_packages: Sequence[str], **kwargs: Any):
        """
        :param log_packages: Names of packages which logs to include.
        """
        super().__init__(*args, **kwargs)

        self.log_packages = _parse_packages(log_packages, self.DEFAULT_LOG_PACKAGES)

    def filter(self, record: LogRecord) -> bool:
        """Shows only logs from eo-grow type packages and high-importance logs"""
        return any(package_name in record.name for package_name in self.log_packages)


class LogFileFilter(Filter):
    """Filters log messages passed to log file"""

    DEFAULT_IGNORE_PACKAGES = (
        "eolearn.core",
        "botocore",
        "s3transfer",
        "matplotlib",
        "fiona",
        "rasterio",
        "graphviz",
        "urllib3",
        "boto3",
    )

    def __init__(self, *args: Any, ignore_packages: Sequence[str], **kwargs: Any):
        """
        :param ignore_packages: Names of packages which logs will be ignored.
        """
        super().__init__(*args, **kwargs)

        self.ignore_packages = _parse_packages(ignore_packages, self.DEFAULT_IGNORE_PACKAGES)

    def filter(self, record: LogRecord) -> bool:
        """Shows everything from the main thread and process except logs from packages that are on the ignore list.
        Those packages send a lot of useless logs.
        """
        if record.name.startswith(self.ignore_packages):
            return False

        return record.threadName == "MainThread" and record.processName == "MainProcess"


class EOExecutionFilter(Filter):
    """Filters logs that will be saved by EOExecutor"""

    DEFAULT_IGNORE_PACKAGES = (
        "botocore",
        "s3transfer",
        "urllib3",
        "rasterio",
        "numba",
        "fiona.ogrext",
        "fiona.env",
        "fiona._env",
    )

    def __init__(self, *args: Any, ignore_packages: Sequence[str], **kwargs: Any):
        """
        :param ignore_packages: Names of packages which logs will be ignored.
        """
        super().__init__(*args, **kwargs)

        self.ignore_packages = _parse_packages(ignore_packages, self.DEFAULT_IGNORE_PACKAGES)

    def filter(self, record: LogRecord) -> bool:
        """Ignores logs from certain low-level packages"""
        return not record.name.startswith(self.ignore_packages)


def _parse_packages(log_packages: Sequence[str], default_packages: Sequence[str]) -> tuple[str, ...]:
    """Builds a tuple of unique package names to be logged/ignored. If the placeholder for default packages is
    present, those are added as well.

    Since many handlers use the `startswith` method we return a tuple (a better alternative is frozenset).
    """
    unique_packages = set(log_packages)
    if DEFAULT_PACKAGES_TOKEN in unique_packages:
        unique_packages.remove(DEFAULT_PACKAGES_TOKEN)
        unique_packages.update(default_packages)
    return tuple(unique_packages)
