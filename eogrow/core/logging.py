"""
Module with utilities for logging
"""
import json
import logging
import sys
import time
from logging import Handler, StreamHandler, FileHandler, Formatter, Filter, LogRecord
from typing import Optional, List, Union, Sequence

import fs
from fs.errors import FilesystemClosed
from pydantic import Field
from sentinelhub import SHConfig
from eolearn.core.utils.fs import join_path

from .base import EOGrowObject
from .config import Config
from .schemas import ManagerSchema
from .storage import StorageManager
from ..utils.fs import LocalFile
from ..utils.general import jsonify
from ..utils.logging import get_instance_info
from ..utils.meta import get_package_versions


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
        pipeline_ignore_packages: Optional[List[str]] = Field(
            description=(
                "Names of packages which logs will not be written to the main pipeline log file. The default null "
                "value means that a default list of packages will be used."
            )
        )
        pipeline_logs_backup_interval: float = Field(
            60,
            description=(
                "When working with a remote storage this parameter defines a minimal number of seconds between "
                "two consecutive times when pipeline log file will be copied into the remote storage."
            ),
        )
        eoexecution_ignore_packages: Optional[List[str]] = Field(
            description=(
                "Names of packages which logs will not be written to EOExecution log files. The default null value "
                "means that a default list of packages will be used."
            )
        )
        include_logs_to_report: bool = Field(
            False,
            description=(
                "If log files should be parsed into an EOExecution report file or just linked. When working "
                "with larger number of EOPatches the recommended option is False."
            ),
        )

        show_logs: bool = Field(False, description="Shows basic pipeline execution logs at stdout.")
        stdout_log_packages: Optional[List[str]] = Field(
            description=(
                "Names of packages which logs will be written to stdout. The default null value means that a default "
                "list of packages will be used."
            )
        )

        capture_warnings: bool = Field(
            True,
            description=(
                "If warnings should be treated as logs and with save_logs=True written into log files instead of "
                "being printed in stderr."
            ),
        )

    def __init__(self, config: Config, storage: StorageManager):
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

    def start_logging(self, pipeline_execution_name: str) -> List[Handler]:
        """Creates a folder for logs and sets up (and returns) logging handlers

        Supported handlers:
        - Writing to a file in pipeline logs folder
        - Printing logs to a standard output
        """
        if self.config.save_logs:
            logs_folder = self.get_pipeline_logs_folder(pipeline_execution_name)
            self.storage.filesystem.makedirs(logs_folder, recreate=True)

        global_logger = logging.getLogger()
        global_logger.setLevel(logging.DEBUG)

        for default_handler in global_logger.handlers:
            default_handler.setLevel(logging.WARNING)

        new_handlers: List[Handler] = []

        if self.config.save_logs:
            file_handler = self._create_file_handler(pipeline_execution_name)
            global_logger.addHandler(file_handler)
            new_handlers.append(file_handler)

        if self.config.show_logs:
            stdout_handler = self._create_stdout_handler()
            global_logger.addHandler(stdout_handler)
            new_handlers.append(stdout_handler)

        if self.config.capture_warnings:
            logging.captureWarnings(True)

        return new_handlers

    def _create_file_handler(self, pipeline_execution_name: str) -> Handler:
        """Creates a logging handler to write a pipeline log to a file."""
        logs_filename = fs.path.combine(self.get_pipeline_logs_folder(pipeline_execution_name), "pipeline.log")
        file_handler = RegularBackupHandler(
            logs_filename,
            filesystem=self.storage.filesystem,
            backup_interval=self.config.pipeline_logs_backup_interval,
            encoding="utf-8",
        )

        formatter = Formatter(
            "%(levelname)s %(asctime)s %(name)s:%(lineno)d:\n\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        file_handler.addFilter(LogFileFilter())

        return file_handler

    def _create_stdout_handler(self) -> Handler:
        """Creates a logging handler to write logs into a standard output."""
        stdout_handler = StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)

        formatter = Formatter("%(levelname)s %(name)s:%(lineno)d: %(message)s")
        stdout_handler.setFormatter(formatter)

        stdout_handler.addFilter(StdoutFilter(log_packages=self.config.stdout_log_packages))

        return stdout_handler

    def stop_logging(self, handlers: List[Handler]):
        """Updates logs, removes pipeline handlers from the global logger and puts global logging level back to
        default
        """
        if self.config.capture_warnings:
            logging.captureWarnings(False)

        global_logger = logging.getLogger()
        for handler in handlers:
            handler.close()
            global_logger.removeHandler(handler)

        global_logger.setLevel(logging.WARNING)

    def update_pipeline_report(
        self,
        pipeline_execution_name: str,
        pipeline_config: Config,
        pipeline_id: str,
        pipeline_timestamp: str,
        elapsed_time: Optional[float] = None,
    ):
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
            "config_parameters": pipeline_config,
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

    def save_eopatch_execution_status(self, pipeline_execution_name: str, finished: list, failed: list):
        """Saves lists of EOPatch names for which execution either finished successfully or failed"""
        if not self.config.save_logs:
            return

        filesystem = self.storage.filesystem
        logs_folder = self.get_pipeline_logs_folder(pipeline_execution_name)

        for eopatches, filename in [(finished, "finished.json"), (failed, "failed.json")]:
            path = fs.path.combine(logs_folder, filename)
            with filesystem.open(path, "w") as file:
                json.dump(eopatches, file, indent=2)


class FilesystemHandler(FileHandler):
    """A filesystem abstraction of FileHandler

    In case the handler gets a local path it behaves the same as FileHandler. In case it gets a remote path it writes
    logs first to a local path and then copies them to the remote location.
    """

    def __init__(self, path: str, filesystem: Optional[fs.base.FS] = None, config: Optional[SHConfig] = None, **kwargs):
        """
        :param path: A path to a log file. It should be an absolute path if filesystem object is not given and relative
            otherwise.
        :param filesystem: A filesystem to where logs will be written.
        :param config: A config object holding credentials.
        :param kwargs: Keyword arguments that will be propagated to FileHandler.
        """
        self.local_file = LocalFile(path, mode="w", filesystem=filesystem, config=config)

        super().__init__(self.local_file.path, **kwargs)

    def close(self) -> None:
        """Closes logging and closes the local file"""
        super().close()
        try:
            self.local_file.close()
        except FilesystemClosed:
            pass


class RegularBackupHandler(FilesystemHandler):
    """A customized FilesystemHandler that makes a copy to a remote location regularly after given amount of time.

    IMPORTANT: Make sure to combine this handler with a correct log filter. More in docstring of emit method.
    """

    def __init__(self, *args, backup_interval: Union[float, int], **kwargs):
        """
        :param backup_interval: A minimal number of seconds before handler will backup the log file to the remote
            location. The backup will only happen when the next log record will be emitted.
        """
        super().__init__(*args, **kwargs)

        self.backup_interval = backup_interval
        self._last_backup_time = time.monotonic()

    def emit(self, record: LogRecord) -> None:
        """Save a new record and backup to remote if the backup hasn't been done in the given amount of time.

        IMPORTANT: Any new log produced in this method must not reach this handler again. Otherwise the process will
        get stuck waiting for a thread lock release. Make sure that you combine this handler with a Filter class that
        filters out logs from botocore and s3transfer!
        """
        super().emit(record)

        if time.monotonic() > self._last_backup_time + self.backup_interval:
            self.local_file.copy_to_remote()
            self._last_backup_time = time.monotonic()


class EOExecutionHandler(FilesystemHandler):
    """A customized FilesystemHandler that makes a copy to a remote location every time a new node in a workflow
    is started.

    IMPORTANT: Make sure to combine this handler with a correct log filter. More in docstring of emit method.
    """

    def emit(self, record: LogRecord) -> None:
        """Save a new record. In case a new node in EOWorkflow is started it will copy the log file to remote.

        IMPORTANT: Any new log produced in this method must not reach this handler again. Otherwise the process will
        get stuck waiting for a thread lock release. Therefore, make sure that you combine this handler with a Filter
        class that filters out logs from botocore, s3transfer and anything else that could be possibly logged during
        `LocalFile.copy_to_remote` call!
        """
        super().emit(record)

        if record.name == "eolearn.core.eoworkflow" and record.message.startswith("Computing"):
            self.local_file.copy_to_remote()


class StdoutFilter(Filter):
    """Filters log messages passed to standard output"""

    DEFAULT_LOG_PACKAGES = (
        "eogrow",
        "__main__",
        "root",
        "sentinelhub.sentinelhub_batch",
    )

    def __init__(self, log_packages: Optional[Sequence[str]] = None, *args, **kwargs):
        """
        :param log_packages: Names of packages which logs to include.
        """
        super().__init__(*args, **kwargs)

        self.log_packages = self.DEFAULT_LOG_PACKAGES if log_packages is None else log_packages

    def filter(self, record: LogRecord) -> bool:
        """Shows only logs from eo-grow type packages and high-importance logs"""
        if record.levelno >= logging.WARNING:
            return True

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

    def __init__(self, ignore_packages: Optional[Sequence[str]] = None, *args, **kwargs):
        """
        :param ignore_packages: Names of packages which logs will be ignored.
        """
        super().__init__(*args, **kwargs)

        self.ignore_packages = self.DEFAULT_IGNORE_PACKAGES if ignore_packages is None else ignore_packages

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
    )

    def __init__(self, ignore_packages: Optional[Sequence[str]] = None, *args, **kwargs):
        """
        :param ignore_packages: Names of packages which logs will be ignored.
        """
        super().__init__(*args, **kwargs)

        self.ignore_packages = self.DEFAULT_IGNORE_PACKAGES if ignore_packages is None else ignore_packages

    def filter(self, record: LogRecord) -> bool:
        """Ignores logs from certain low-level packages"""
        if record.levelno >= logging.INFO:
            return True

        return not record.name.startswith(self.ignore_packages)
