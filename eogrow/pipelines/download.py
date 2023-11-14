"""Implements different customizeable pipelines for downloading data."""

from __future__ import annotations

import abc
import datetime as dt
import logging
from typing import Any, Callable, List, Optional, Tuple

import fs
import numpy as np
import ray
from pydantic import Field

from eolearn.core import EONode, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.core.types import Feature
from eolearn.io import SentinelHubDemTask, SentinelHubEvalscriptTask, SentinelHubInputTask
from sentinelhub import (
    Band,
    DataCollection,
    MosaickingOrder,
    ResamplingType,
    SentinelHubSession,
    Unit,
)

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..tasks.common import LinearFunctionTask
from ..types import ExecKwargs, PatchList, TimePeriod
from ..utils.filter import get_patches_with_missing_features
from ..utils.validators import (
    ensure_exactly_one_defined,
    ensure_storage_key_presence,
    field_validator,
    optional_field_validator,
    parse_data_collection,
    parse_dtype,
    parse_time_period,
)

LOGGER = logging.getLogger(__name__)

SessionLoaderType = Optional[Callable[[], SentinelHubSession]]


@ray.remote
class RaySessionActor:
    """This object can share a session object with multiple workers."""

    def __init__(self, session: SentinelHubSession):
        self.session = session

    def get_valid_session(self) -> SentinelHubSession:
        """The following makes sure that a token is still valid or refreshed, and returns it in a
        non-refreshing session object."""
        token = self.session.token
        return SentinelHubSession.from_token(token)


class RescaleSchema(BaseSchema):
    rescale_factor: float = Field(1, description="Amount by which the selected features are multiplied")
    dtype: Optional[np.dtype] = Field(description="The output dtype of data")
    _parse_dtype = optional_field_validator("dtype", parse_dtype, pre=True)
    features_to_rescale: List[Feature]


class PostprocessingRescale(BaseSchema):
    rescale_schemas: List[RescaleSchema] = Field(
        default_factory=list, description="Specify different ways to rescale features"
    )


class BaseDownloadPipeline(Pipeline, metaclass=abc.ABCMeta):
    """Base pipeline for downloading satellite data"""

    class Schema(Pipeline.Schema):
        output_folder_key: str = Field(
            description="Storage manager key pointing to the path where downloaded EOPatches will be saved."
        )
        _ensure_output_folder_key = ensure_storage_key_presence("output_folder_key")

        threads_per_worker: Optional[int] = Field(
            description=(
                "Maximum number of parallel threads used during download by each worker. If set to None it will use "
                "5 * N threads, where N is the number of CPUs on the machine"
            ),
        )

        postprocessing: Optional[PostprocessingRescale] = Field(description="Parameters used in post-processing tasks")

    config: Schema

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.download_node_uid = "<NODE ID NOT SET>"

    def filter_patch_list(self, patch_list: PatchList) -> PatchList:
        """EOPatches are filtered according to existence of specified output features"""
        output_features = self._get_output_features()
        return get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            output_features,
            check_timestamps=any(ftype.is_temporal() for ftype, _ in output_features),
        )

    @abc.abstractmethod
    def _get_output_features(self) -> list[Feature]:
        """Lists all features that are to be saved upon the pipeline completion"""

    @abc.abstractmethod
    def _get_download_node(self, session_loader: SessionLoaderType) -> EONode:
        """Provides node for downloading data."""

    def _create_session_loader(self) -> SessionLoaderType:
        session = SentinelHubSession(self.sh_config)
        actor = RaySessionActor.remote(session)  # type: ignore[attr-defined]
        return lambda: ray.get(actor.get_valid_session.remote())

    @staticmethod
    def get_postprocessing_node(postprocessing_config: PostprocessingRescale, previous_node: EONode) -> EONode:
        """Provides node that applies postprocessing to data after download is complete"""
        node = previous_node
        for rescale_config in postprocessing_config.rescale_schemas:
            node = EONode(
                LinearFunctionTask(
                    rescale_config.features_to_rescale, slope=rescale_config.rescale_factor, dtype=rescale_config.dtype
                ),
                inputs=[node],
            )

        return node

    def build_workflow(self, session_loader: SessionLoaderType) -> EOWorkflow:
        """Method that builds a workflow"""
        download_node = self._get_download_node(session_loader)
        self.download_node_uid = download_node.uid

        postprocessing_node = None
        if self.config.postprocessing:
            postprocessing_node = self.get_postprocessing_node(self.config.postprocessing, download_node)

        end_node = EONode(
            SaveTask(
                self.storage.get_folder(self.config.output_folder_key),
                filesystem=self.storage.filesystem,
                features=self._get_output_features(),
                overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                use_zarr=self.storage.config.use_zarr,
            ),
            inputs=[postprocessing_node or download_node],
        )

        return EOWorkflow.from_endnodes(end_node)

    def get_execution_arguments(self, workflow: EOWorkflow, patch_list: PatchList) -> ExecKwargs:
        """Adds required bbox and time_interval parameters for input task to the base execution arguments

        :param workflow: EOWorkflow used to download images
        """
        exec_args = super().get_execution_arguments(workflow, patch_list)

        download_node = workflow.get_node_with_uid(self.download_node_uid, fail_if_missing=True)

        for patch_name, bbox in patch_list:
            exec_args[patch_name][download_node] = {"bbox": bbox}
            if hasattr(self.config, "time_period"):
                exec_args[patch_name][download_node]["time_interval"] = self.config.time_period

        return exec_args

    def run_procedure(self) -> tuple[list[str], list[str]]:
        patch_list = self.get_patch_list()
        workflow = self.build_workflow(self._create_session_loader())
        exec_args = self.get_execution_arguments(workflow, patch_list)

        finished, failed, _ = self.run_execution(workflow, exec_args)

        return finished, failed


class CommonDownloadFields(BaseSchema):
    data_collection: DataCollection = Field(
        description=(
            "Data collection from which data will be downloaded. See `utils.validators.parse_data_collection` for more"
            " info on input options."
        )
    )
    _validate_data_collection = field_validator("data_collection", parse_data_collection, pre=True)

    resolution: Optional[float] = Field(
        description=(
            "Resolution of downloaded data in meters. Exactly one of the parameters resolution and size has to be"
            " specified."
        )
    )
    size: Optional[Tuple[int, int]] = Field(
        description=(
            "A pair (width, height) of downloaded data in pixels. . Exactly one of the parameters resolution and size"
            " has to be specified."
        )
    )

    maxcc: Optional[float] = Field(ge=0, le=1, description="Maximal cloud coverage filter.")

    resampling_type: Optional[ResamplingType] = Field(
        description="A type of downsampling and upsampling used by Sentinel Hub service. Default is NEAREST"
    )

    _check_resolution_and_size = ensure_exactly_one_defined("resolution", "size")


class TimeDependantFields(BaseSchema):
    time_period: TimePeriod
    _validate_time_period = field_validator("time_period", parse_time_period, pre=True)

    time_difference: Optional[float] = Field(description="Time difference in minutes between consecutive time frames")

    mosaicking_order: Optional[MosaickingOrder] = Field(
        description="The mosaicking order used by Sentinel Hub service. Default is mostRecent"
    )


class DownloadPipeline(BaseDownloadPipeline):
    """Pipeline to download data via SentinelHubInputTask"""

    class Schema(BaseDownloadPipeline.Schema, CommonDownloadFields, TimeDependantFields):
        bands_feature_name: str = Field(description="Name of a feature in which bands will be saved")
        bands: Optional[List[str]] = Field(description="Names of bands to download")
        additional_data: List[Feature] = Field(default_factory=list, description="Additional data to download")
        use_dn: bool = Field(
            False, description="Save bands as float32 reflectance (default), or int16 digital numbers."
        )

    config: Schema

    def _get_output_features(self) -> list[Feature]:
        return [(FeatureType.DATA, self.config.bands_feature_name), *self.config.additional_data]

    def _get_download_node(self, session_loader: SessionLoaderType) -> EONode:
        time_diff = None if self.config.time_difference is None else dt.timedelta(minutes=self.config.time_difference)
        bands_dtype = np.uint16 if self.config.use_dn else np.float32
        data_collection = self.config.data_collection

        if data_collection.is_byoc:
            if not self.config.bands:
                raise ValueError("Band names must be explicitly supplied when working with BYOC.")

            data_collection = self.config.data_collection.define_from(
                f"{self.config.data_collection.name}_WITH_BANDS",
                bands=[Band(name=band, units=(Unit.DN,), output_types=(bands_dtype,)) for band in self.config.bands],
                metabands=[(Band(name="dataMask", units=(Unit.DN,), output_types=(bool,)))],
            )

        download_task = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, self.config.bands_feature_name),
            bands=self.config.bands,
            resolution=self.config.resolution,
            size=self.config.size,
            maxcc=self.config.maxcc,
            time_difference=time_diff,
            data_collection=data_collection,
            max_threads=self.config.threads_per_worker,
            additional_data=self.config.additional_data,
            bands_dtype=bands_dtype,
            config=self.sh_config,
            mosaicking_order=self.config.mosaicking_order,
            downsampling=self.config.resampling_type,
            upsampling=self.config.resampling_type,
            session_loader=session_loader,
        )
        return EONode(download_task)


class DownloadEvalscriptPipeline(BaseDownloadPipeline):
    """Pipeline to download through an evalscript"""

    class Schema(BaseDownloadPipeline.Schema, CommonDownloadFields, TimeDependantFields):
        features: List[Feature] = Field(description="Features to construct from the evalscript")
        evalscript_folder_key: str = Field(
            "input_data", description="Storage manager key pointing to the path where the evalscript is loaded from."
        )
        _ensure_evalscript_folder_key = ensure_storage_key_presence("evalscript_folder_key")
        evalscript_path: str

    config: Schema

    def _get_output_features(self) -> list[Feature]:
        return self.config.features

    def _get_download_node(self, session_loader: SessionLoaderType) -> EONode:
        evalscript = self._get_evalscript()

        time_diff = None if self.config.time_difference is None else dt.timedelta(minutes=self.config.time_difference)

        download_task = SentinelHubEvalscriptTask(
            features=self.config.features,
            evalscript=evalscript,
            data_collection=self.config.data_collection,
            resolution=self.config.resolution,
            size=self.config.size,
            maxcc=self.config.maxcc,
            time_difference=time_diff,
            max_threads=self.config.threads_per_worker,
            config=self.sh_config,
            mosaicking_order=self.config.mosaicking_order,
            downsampling=self.config.resampling_type,
            upsampling=self.config.resampling_type,
            session_loader=session_loader,
        )
        return EONode(download_task)

    def _get_evalscript(self) -> str:
        evalscript_path = fs.path.join(
            self.storage.get_folder(self.config.evalscript_folder_key), self.config.evalscript_path
        )
        with self.storage.filesystem.open(evalscript_path) as evalscript_file:
            return evalscript_file.read()


class DownloadTimelessPipeline(BaseDownloadPipeline):
    """Pipeline to download timeless data"""

    class Schema(BaseDownloadPipeline.Schema, CommonDownloadFields):
        feature_name: str = Field(description="Name of the resulting feature")

    config: Schema

    def _get_output_features(self) -> list[Feature]:
        return [(FeatureType.DATA_TIMELESS, self.config.feature_name)]

    def _get_download_node(self, session_loader: SessionLoaderType) -> EONode:
        download_task = SentinelHubDemTask(
            feature=(FeatureType.DATA_TIMELESS, self.config.feature_name),
            data_collection=self.config.data_collection,
            resolution=self.config.resolution,
            size=self.config.size,
            maxcc=self.config.maxcc,
            max_threads=self.config.threads_per_worker,
            config=self.sh_config,
            downsampling=self.config.resampling_type,
            upsampling=self.config.resampling_type,
            session_loader=session_loader,
        )
        return EONode(download_task)
