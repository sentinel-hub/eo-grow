"""
Module implementing pipelines for downloading data
"""
import abc
import datetime as dt
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field

from eolearn.core import EONode, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.features import LinearFunctionTask
from eolearn.io import SentinelHubDemTask, SentinelHubEvalscriptTask, SentinelHubInputTask
from sentinelhub import Band, DataCollection, MimeType, MosaickingOrder, ResamplingType, Unit, read_data

from ..core.pipeline import Pipeline
from ..core.schemas import BaseSchema
from ..utils.filter import get_patches_with_missing_features
from ..utils.types import Feature, FeatureSpec, Path, TimePeriod
from ..utils.validators import field_validator, parse_data_collection, parse_time_period

LOGGER = logging.getLogger(__name__)


class RescaleSchema(BaseSchema):
    rescale_factor: float = Field(1, description="Amount by which the selected features are multiplied")
    dtype: Optional[str] = Field(description="The output dtype of data")
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

        compress_level: int = Field(1, description="Level of compression used in saving EOPatches")
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
        self.download_node_uid: Optional[str] = None

    def filter_patch_list(self, patch_list: List[str]) -> List[str]:
        """EOPatches are filtered according to existence of specified output features"""

        filtered_patch_list = get_patches_with_missing_features(
            self.storage.filesystem,
            self.storage.get_folder(self.config.output_folder_key),
            patch_list,
            self._get_output_features(),
        )

        return filtered_patch_list

    @abc.abstractmethod
    def _get_output_features(self) -> List[FeatureSpec]:
        """Lists all features that are to be saved upon the pipeline completion"""

    @abc.abstractmethod
    def _get_download_node(self) -> EONode:
        """Provides node for downloading data."""

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

    def build_workflow(self) -> EOWorkflow:
        """Method that builds a workflow"""
        download_node = self._get_download_node()
        self.download_node_uid = download_node.uid

        postprocessing_node = None
        if self.config.postprocessing:
            postprocessing_node = self.get_postprocessing_node(self.config.postprocessing, download_node)

        end_node = EONode(
            SaveTask(
                self.storage.get_folder(self.config.output_folder_key, full_path=True),
                features=self._get_output_features(),
                compress_level=self.config.compress_level,
                overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
                config=self.sh_config,
            ),
            inputs=[postprocessing_node or download_node],
        )

        return EOWorkflow.from_endnodes(end_node)

    def get_execution_arguments(self, workflow: EOWorkflow) -> List[Dict[EONode, Dict[str, object]]]:
        """Adds required bbox and time_interval parameters for input task to the base execution arguments

        :param workflow: EOWorkflow used to download images
        """
        exec_args = super().get_execution_arguments(workflow)

        download_node = workflow.get_node_with_uid(self.download_node_uid)
        if download_node is None:
            return exec_args

        bbox_list = self.eopatch_manager.get_bboxes(eopatch_list=self.patch_list)

        for index, bbox in enumerate(bbox_list):
            exec_args[index][download_node] = {"bbox": bbox}
            if hasattr(self.config, "time_period"):
                exec_args[index][download_node]["time_interval"] = self.config.time_period  # type: ignore

        return exec_args


class CommonDownloadFields(BaseSchema):
    data_collection: DataCollection = Field(description="Data collection from which data will be downloaded.")
    _validate_data_collection = field_validator("data_collection", parse_data_collection, pre=True)

    resolution: float = Field(description="Resolution of downloaded data in meters")

    maxcc: Optional[float] = Field(ge=0, le=1, description="Maximal cloud coverage filter.")

    resampling_type: Optional[ResamplingType] = Field(
        description="A type of downsampling and upsampling used by Sentinel Hub service. Default is NEAREST"
    )


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
            False, description="Whether to save bands as float32 reflectance (default), or int16 digital numbers."
        )

    config: Schema

    def _get_output_features(self) -> List[FeatureSpec]:
        features: List[FeatureSpec] = [
            (FeatureType.DATA, self.config.bands_feature_name),
            FeatureType.BBOX,
            FeatureType.TIMESTAMP,
        ]
        features.extend(self.config.additional_data)
        return features

    def _get_download_node(self) -> EONode:
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
            maxcc=self.config.maxcc,
            time_difference=time_diff,
            data_collection=data_collection,
            max_threads=self.config.threads_per_worker,
            additional_data=self.config.additional_data,
            bands_dtype=bands_dtype,
            config=self.sh_config,
            mosaicking_order=self.config.mosaicking_order,
            aux_request_args=_get_aux_request_args(self.config.resampling_type),
        )
        return EONode(download_task)


class DownloadEvalscriptPipeline(BaseDownloadPipeline):
    """Pipeline to download through an evalscript"""

    class Schema(BaseDownloadPipeline.Schema, CommonDownloadFields, TimeDependantFields):
        features: List[Feature] = Field(description="Features to construct from the evalscript")
        evalscript_path: Path

    config: Schema

    def _get_output_features(self) -> List[FeatureSpec]:
        features: List[FeatureSpec] = [FeatureType.BBOX, FeatureType.TIMESTAMP]
        features.extend(self.config.features)
        return features

    def _get_download_node(self) -> EONode:
        evalscript = read_data(self.config.evalscript_path, data_format=MimeType.TXT)
        time_diff = None if self.config.time_difference is None else dt.timedelta(minutes=self.config.time_difference)

        download_task = SentinelHubEvalscriptTask(
            features=self.config.features,
            evalscript=evalscript,
            data_collection=self.config.data_collection,
            resolution=self.config.resolution,
            maxcc=self.config.maxcc,
            time_difference=time_diff,
            max_threads=self.config.threads_per_worker,
            config=self.sh_config,
            mosaicking_order=self.config.mosaicking_order,
            aux_request_args=_get_aux_request_args(self.config.resampling_type),
        )
        return EONode(download_task)


class DownloadTimelessPipeline(BaseDownloadPipeline):
    """Pipeline to download timeless data"""

    class Schema(BaseDownloadPipeline.Schema, CommonDownloadFields):
        feature_name: str = Field(description="Name of the resulting feature")

    config: Schema

    def _get_output_features(self) -> List[FeatureSpec]:
        return [(FeatureType.DATA_TIMELESS, self.config.feature_name), FeatureType.BBOX]

    def _get_download_node(self) -> EONode:

        download_task = SentinelHubDemTask(
            feature=(FeatureType.DATA_TIMELESS, self.config.feature_name),
            data_collection=self.config.data_collection,
            resolution=self.config.resolution,
            maxcc=self.config.maxcc,
            max_threads=self.config.threads_per_worker,
            config=self.sh_config,
            aux_request_args=_get_aux_request_args(self.config.resampling_type),
        )
        return EONode(download_task)


def _get_aux_request_args(resampling: Optional[ResamplingType]) -> Optional[dict]:
    if resampling is not None:
        return {"processing": {"downsampling": resampling, "upsampling": resampling}}
    return None
