"""Implementation of the base Pipeline class."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, TypeVar

import ray

from eolearn.core import CreateEOPatchTask, EOExecutor, EONode, EOWorkflow, LoadTask, SaveTask, WorkflowResults
from eolearn.core.extra.ray import RayExecutor

from ..types import ExecKwargs, PatchList
from ..utils.general import current_timestamp
from ..utils.meta import import_object
from .area.base import BaseAreaManager
from .base import EOGrowObject
from .config import RawConfig
from .logging import EOExecutionFilter, EOExecutionHandler, LoggingManager
from .schemas import ManagerSchema, PipelineSchema
from .storage import StorageManager

Self = TypeVar("Self", bound="Pipeline")

LOGGER = logging.getLogger(__name__)


class Pipeline(EOGrowObject):
    """A base class for execution of processing procedures which may or may not include running EOWorkflows, running
    EOExecutions, creating maps, etc.

    The functionalities of this class are:
        - collecting input arguments (either from command line or as an initialization parameter) and parsing them
        - preparing a list of patches
        - preparing execution arguments
        - running the pipeline, monitoring, and reporting
    """

    class Schema(PipelineSchema):
        """Configuration schema, describing input parameters and their types."""

    config: Schema

    def __init__(self, config: Schema, raw_config: RawConfig | None = None):
        """
        :param config: A dictionary with configuration parameters
        :param raw_config: The configuration parameters pre-validation, for logging purposes only
        """
        super().__init__(config)
        self._raw_config = raw_config

        self.pipeline_id = self._new_pipeline_id()
        self.current_execution_name = "<Not executed yet>"

        self.storage: StorageManager = self._load_manager(config.storage)
        self.sh_config = self.storage.sh_config

        self.area_manager: BaseAreaManager = self._load_manager(config.area, storage=self.storage)
        self.logging_manager: LoggingManager = self._load_manager(config.logging, storage=self.storage)

    @property
    def _pipeline_name(self) -> str:
        return self.config.pipeline_name or self.__class__.__name__

    @classmethod
    def from_raw_config(cls: type[Self], config: RawConfig, *args: Any, **kwargs: Any) -> Self:
        """Creates an object from a dictionary by constructing a validated config and use it to create the object."""
        validated_config = cls.Schema.parse_obj(config)
        if "raw_config" not in kwargs:
            kwargs["raw_config"] = config
        return cls(validated_config, *args, **kwargs)

    @staticmethod
    def _new_pipeline_id() -> str:
        """Provides a new random uuid of a pipeline"""
        return uuid.uuid4().hex[:10]

    @staticmethod
    def _load_manager(manager_config: ManagerSchema, **manager_params: Any) -> Any:
        """Loads a manager class and back-propagates parsed config

        :param manager_config: A sub-config with manager parameters
        :param manager_params: Other parameters to initialize a manager class
        """
        if manager_config.manager is None:
            raise ValueError("Unable to load manager, field `manager` specifying it's class is missing.")
        manager_class = import_object(manager_config.manager)
        return manager_class(manager_config, **manager_params)

    def get_pipeline_execution_name(self, pipeline_timestamp: str) -> str:
        """Returns the full name of the pipeline execution"""
        return f"{pipeline_timestamp}-{self._pipeline_name}-{self.pipeline_id}"

    def get_patch_list(self) -> PatchList:
        """Method that prepares the list of EOPatches for which to run the pipeline execution."""
        patch_list = self.area_manager.get_patch_list()

        if self.config.test_subset is not None:
            LOGGER.info("Filtering according to `test_subset` parameter.")
            indices = {x for x in self.config.test_subset if isinstance(x, int)}
            names = {x for x in self.config.test_subset if isinstance(x, str)}
            patch_list = [(name, bbox) for i, (name, bbox) in enumerate(patch_list) if (i in indices or name in names)]

            if len(patch_list) < len(self.config.test_subset):
                raise ValueError(
                    f"The parameter `test_subset` specifies {len(self.config.test_subset)} patches, but only"
                    f" {len(patch_list)} remain after filtration. Please recheck your input for `test_subset`."
                )

        if self.config.skip_existing:
            LOGGER.info("Checking which EOPatches can be skipped")
            filtered_patch_list = self.filter_patch_list(patch_list)

            skip_message = (
                "Skipped some EOPatches" if len(filtered_patch_list) < len(patch_list) else "No EOPatches were skipped"
            )
            LOGGER.info("%s, %d / %d remaining", skip_message, len(filtered_patch_list), len(patch_list))

            return filtered_patch_list

        return patch_list

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """Specifies which EOPatches should be skipped when `skip_existing` is enabled."""
        raise NotImplementedError("Method `filter_patch_list` must be implemented in order to use `skip_existing`")

    def get_execution_arguments(self, workflow: EOWorkflow, patch_list: PatchList) -> ExecKwargs:
        """Prepares execution arguments for each eopatch from a list of patches.

        The output should be a dictionary of form `{execution_name: {node: node_kwargs}}`. Execution names are usually
        names of EOPatches, but can be anything.

        :param workflow: A workflow for which arguments will be prepared
        """
        exec_kwargs = {}
        nodes = workflow.get_nodes()
        for name, bbox in patch_list:
            patch_args: dict[EONode, dict[str, Any]] = {}

            for node in nodes:
                if isinstance(node.task, (SaveTask, LoadTask)):
                    patch_args[node] = dict(eopatch_folder=name)

                if isinstance(node.task, CreateEOPatchTask):
                    patch_args[node] = dict(bbox=bbox)

            exec_kwargs[name] = patch_args
        return exec_kwargs

    def run_execution(
        self,
        workflow: EOWorkflow,
        execution_kwargs: ExecKwargs,
        **executor_run_params: Any,
    ) -> tuple[list[str], list[str], list[WorkflowResults]]:
        """A method which runs EOExecutor on given workflow with given execution parameters

        :param workflow: A workflow to be executed
        :param execution_kwargs: A dictionary mapping execution names to dictionaries holding execution arguments
        :param eopatch_list: A custom list of EOPatch names on which execution will run. If not specified, the default
            self.patch_list will be used
        :return: Lists of successfully/unsuccessfully executed EOPatch names and the result of the EOWorkflow execution
        """
        if self.config.debug:
            executor_class: type[EOExecutor] = EOExecutor
            executor_kwargs = {}
        else:
            ray.init(address="auto", ignore_reinit_error=True)
            executor_class = RayExecutor
            executor_kwargs = {"ray_remote_kwargs": self.config.worker_resources}

        LOGGER.info("Starting processing for %d EOPatches", len(execution_kwargs))

        # Unpacking manually to ensure order matches
        list_of_kwargs, execution_names = [], []
        for exec_name, exec_kwargs in execution_kwargs.items():
            list_of_kwargs.append(exec_kwargs)
            execution_names.append(exec_name)

        executor = executor_class(
            workflow,
            list_of_kwargs,
            execution_names=execution_names,
            save_logs=self.logging_manager.config.save_logs,
            logs_folder=self.logging_manager.get_pipeline_logs_folder(self.current_execution_name),
            filesystem=self.storage.filesystem,
            logs_filter=EOExecutionFilter(ignore_packages=self.logging_manager.config.eoexecution_ignore_packages),
            logs_handler_factory=EOExecutionHandler,
            raise_on_temporal_mismatch=self.config.raise_on_temporal_mismatch,
            **executor_kwargs,
        )
        execution_results = executor.run(**executor_run_params)

        successful = [execution_names[idx] for idx in executor.get_successful_executions()]
        failed = [execution_names[idx] for idx in executor.get_failed_executions()]
        LOGGER.info(
            "%s finished with %d / %d success rate",
            executor_class.__name__,
            len(successful),
            len(successful) + len(failed),
        )

        if self.logging_manager.config.save_logs:
            executor.make_report(include_logs=self.logging_manager.config.include_logs_to_report)
            LOGGER.info("Saved EOExecution report to %s", executor.get_report_path(full_path=True))

        return successful, failed, execution_results

    def run(self) -> None:
        """The main method for pipeline execution. It sets up logging and runs the pipeline procedure."""
        timestamp = current_timestamp()
        self.current_execution_name = self.get_pipeline_execution_name(timestamp)

        root_logger = logging.getLogger()
        handlers = self.logging_manager.start_logging(root_logger, self.current_execution_name, "pipeline.log")
        try:
            self.logging_manager.update_pipeline_report(
                pipeline_execution_name=self.current_execution_name,
                pipeline_config=self.config,
                pipeline_raw_config=self._raw_config,
                pipeline_id=self.pipeline_id,
                pipeline_timestamp=timestamp,
            )

            LOGGER.info("Running %s", self._pipeline_name)

            pipeline_start = time.time()
            finished, failed = self.run_procedure()
            elapsed_time = time.time() - pipeline_start

            if failed:
                LOGGER.info(
                    "Pipeline finished with some errors! Check %s",
                    self.logging_manager.get_pipeline_logs_folder(self.current_execution_name, full_path=True),
                )
            else:
                LOGGER.info("Pipeline finished successfully!")

            self.logging_manager.update_pipeline_report(
                pipeline_execution_name=self.current_execution_name,
                pipeline_raw_config=self._raw_config,
                pipeline_config=self.config,
                pipeline_id=self.pipeline_id,
                pipeline_timestamp=timestamp,
                elapsed_time=elapsed_time,
            )

            self.logging_manager.save_eopatch_execution_status(
                pipeline_execution_name=self.current_execution_name, finished=finished, failed=failed
            )
        finally:
            self.logging_manager.stop_logging(root_logger, handlers)

    def run_procedure(self) -> tuple[list[str], list[str]]:
        """Execution procedure of pipeline. Can be overridden if needed.

        By default, builds the workflow by using a `build_workflow` method, which must be additionally implemented.

        :return: A pair of lists representing successful and unsuccessful executions.
        """
        if not hasattr(self, "build_workflow"):
            raise NotImplementedError(
                "Implementation of the `run_procedure` method requires implementation of the `build_workflow` method."
            )
        workflow = self.build_workflow()
        patch_list = self.get_patch_list()
        exec_args = self.get_execution_arguments(workflow, patch_list)

        finished, failed, _ = self.run_execution(workflow, exec_args)
        return finished, failed
