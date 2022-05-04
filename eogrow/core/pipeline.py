"""
Module where base Pipeline class is implemented
"""
import datetime as dt
import functools
import logging
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import ray

from eolearn.core import CreateEOPatchTask, EOExecutor, EONode, EOWorkflow, LoadTask, SaveTask, WorkflowResults
from eolearn.core.extra.ray import RayExecutor

from ..utils.meta import import_object
from .area.base import AreaManager
from .base import EOGrowObject
from .config import RawConfig
from .eopatch import EOPatchManager
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
        - monitoring the pipeline and reporting
    """

    class Schema(PipelineSchema):
        """Configuration schema, describing input parameters and their types."""

    config: Schema

    def __init__(self, config: Schema, raw_config: Optional[RawConfig] = None):
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

        self.area_manager: AreaManager = self._load_manager(config.area, storage=self.storage)
        self.eopatch_manager: EOPatchManager = self._load_manager(config.eopatch, area_manager=self.area_manager)
        self.logging_manager: LoggingManager = self._load_manager(config.logging, storage=self.storage)

        self._patch_list: Optional[List[str]] = None

    @classmethod
    def from_raw_config(cls: Type[Self], config: RawConfig, *args: Any, **kwargs: Any) -> Self:
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

        :param manager_key: A config key name of a sub-config with manager parameters
        :param manager_params: Other parameters to initialize a manager class
        """
        if manager_config.manager is None:
            raise ValueError("Unable to load manager, field `manager` specifying it's class is missing.")
        manager_class = import_object(manager_config.manager)
        manager = manager_class(manager_config, **manager_params)
        return manager

    def get_pipeline_execution_name(self, pipeline_timestamp: str) -> str:
        """Returns the full name of the pipeline execution"""
        return f"{pipeline_timestamp}-{self.__class__.__name__}-{self.pipeline_id}"

    @property
    def patch_list(self) -> List[str]:
        """A property that provides a list of EOPatch names that will be used by the pipeline. The list is calculated
        lazily the first time this property is called.
        """
        if self._patch_list is None:
            self._patch_list = self._prepare_patch_list()
        return self._patch_list

    def _prepare_patch_list(self) -> List[str]:
        """Method which at the initialization prepares the list of EOPatches which will be used"""
        if self.config.input_patch_file is None:
            patch_list = self.eopatch_manager.get_eopatch_filenames(id_list=self.config.patch_list)
        else:
            if self.config.patch_list:
                warnings.warn(
                    "'patch_list' and 'input_patch_file' parameters have both been given, therefore patches "
                    "from input_patch_file will be filtered according to patch_list indices",
                    RuntimeWarning,
                )
            patch_list = self.eopatch_manager.load_eopatch_filenames(
                self.config.input_patch_file, id_list=self.config.patch_list
            )

        if self.config.skip_existing:
            LOGGER.info("Checking which EOPatches can be skipped")
            filtered_patch_list = self.filter_patch_list(patch_list)

            skip_message = (
                "Skipped some EOPatches" if len(filtered_patch_list) < len(patch_list) else "No EOPatches were skipped"
            )
            LOGGER.info("%s, %d / %d remaining", skip_message, len(filtered_patch_list), len(patch_list))

            patch_list = filtered_patch_list

        return patch_list

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """Overwrite this method to specify which EOPatches should be filtered with `skip_existing`"""
        raise NotImplementedError("Method `filter_patch_list` must be implemented in order to use `skip_existing`")

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        """Prepares execution arguments for each eopatch from a list of patches

        :param workflow: A workflow for which arguments will be prepared
        """
        bbox_list = self.eopatch_manager.get_bboxes(eopatch_list=self.patch_list)

        exec_args = []
        nodes = workflow.get_nodes()
        for name, bbox in zip(self.patch_list, bbox_list):
            single_exec_dict: Dict[EONode, Dict[str, Any]] = {}

            for node in nodes:
                if isinstance(node.task, (SaveTask, LoadTask)):
                    single_exec_dict[node] = dict(eopatch_folder=name)

                if isinstance(node.task, CreateEOPatchTask):
                    single_exec_dict[node] = dict(bbox=bbox)

            exec_args.append(single_exec_dict)
        return exec_args

    def run_execution(
        self,
        workflow: EOWorkflow,
        exec_args: List[dict],
        eopatch_list: Optional[List[str]] = None,
        **executor_run_params: Any,
    ) -> Tuple[List[str], List[str], List[WorkflowResults]]:
        """A method which runs EOExecutor on given workflow with given execution parameters

        :param workflow: A workflow to be executed
        :param exec_args: A list of dictionaries holding execution arguments
        :param eopatch_list: A custom list of EOPatch names on which execution will run. If not specified, the default
            self.patch_list will be used
        :return: Lists of successfully/unsuccessfully executed EOPatch names and the result of the EOWorkflow execution
        """
        if eopatch_list is None:
            eopatch_list = self.patch_list

        executor_class: Type[EOExecutor]

        if self.config.use_ray == "auto":
            try:
                LOGGER.info("Searching for Ray cluster")
                ray.init(address="auto", ignore_reinit_error=True)
                executor_class = RayExecutor
                LOGGER.info("Cluster found, pipeline will run using the RayExecutor.")
            except ConnectionError:
                LOGGER.info("No cluster found, pipeline will not use Ray.")
                executor_class = EOExecutor
                executor_run_params["workers"] = self.config.workers
        elif self.config.use_ray:
            ray.init(address="auto", ignore_reinit_error=True)
            executor_class = RayExecutor
        else:
            executor_class = EOExecutor
            executor_run_params["workers"] = self.config.workers

        LOGGER.info("Starting %s for %d EOPatches", executor_class.__name__, len(exec_args))
        executor = executor_class(
            workflow,
            exec_args,
            execution_names=eopatch_list,
            save_logs=self.config.logging.save_logs,
            logs_folder=self.logging_manager.get_pipeline_logs_folder(self.current_execution_name),
            filesystem=self.storage.filesystem,
            logs_filter=EOExecutionFilter(ignore_packages=self.config.logging.eoexecution_ignore_packages),
            logs_handler_factory=functools.partial(EOExecutionHandler, config=self.sh_config, encoding="utf-8"),
        )
        execution_results = executor.run(**executor_run_params)

        successful_eopatches = [eopatch_list[idx] for idx in executor.get_successful_executions()]
        failed_eopatches = [eopatch_list[idx] for idx in executor.get_failed_executions()]
        LOGGER.info(
            "%s finished with %d / %d success rate",
            executor_class.__name__,
            len(successful_eopatches),
            len(successful_eopatches) + len(failed_eopatches),
        )

        if self.config.logging.save_logs:
            executor.make_report(include_logs=self.config.logging.include_logs_to_report)
            LOGGER.info("Saved EOExecution report to %s", executor.get_report_path(full_path=True))

        return successful_eopatches, failed_eopatches, execution_results

    def run(self) -> None:
        """Call this method to run any pipeline"""
        timestamp = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        self.current_execution_name = self.get_pipeline_execution_name(timestamp)

        handlers = self.logging_manager.start_logging(self.current_execution_name)
        try:
            self.logging_manager.update_pipeline_report(
                pipeline_execution_name=self.current_execution_name,
                pipeline_config=self.config,
                pipeline_raw_config=self._raw_config,
                pipeline_id=self.pipeline_id,
                pipeline_timestamp=timestamp,
            )

            LOGGER.info("Running %s", self.__class__.__name__)

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

            finished = self.eopatch_manager.parse_eopatch_list(finished)
            failed = self.eopatch_manager.parse_eopatch_list(failed)
            self.logging_manager.save_eopatch_execution_status(
                pipeline_execution_name=self.current_execution_name, finished=finished, failed=failed
            )
        finally:
            self.logging_manager.stop_logging(handlers)

    def run_procedure(self) -> Tuple[List[str], List[str]]:
        """Execution procedure of pipeline. Can be overridden if needed.

        By default, builds the workflow by using a `build_workflow` method, which must be additionally implemented.

        :return: A list of successfully executed EOPatch names and a list of unsuccessfully executed EOPatch names
        """
        if not hasattr(self, "build_workflow"):
            raise NotImplementedError(
                "Default implementation of run_procedure method requires implementation of build_workflow method"
            )
        workflow = self.build_workflow()  # type: ignore
        exec_args = self.get_execution_arguments(workflow)

        finished, failed, _ = self.run_execution(workflow, exec_args)
        return finished, failed
