""" Includes custom types used in schemas
"""
import datetime
from enum import Enum
from typing import Any, Dict, Literal, Tuple, Union

from eolearn.core import FeatureType

Path = str
S3Path = str
ImportPath = str
TimePeriod = Tuple[datetime.date, datetime.date]

Feature = Tuple[FeatureType, str]
FeatureSpec = Union[Tuple[FeatureType, str], FeatureType]

BoolOrAuto = Union[Literal["auto"], bool]

JsonDict = Dict[str, Any]
RawSchemaDict = Dict[str, Any]

AwsAclType = Literal[
    "private",
    "public-read",
    "public-read-write",
    "aws-exec-read",
    "authenticated-read",
    "bucket-owner-read",
    "bucket-owner-full-control",
    "log-delivery-write",
]


class ProcessingType(Enum):
    RAY = "ray"
    SINGLE = "single"
    MULTI = "multi"
