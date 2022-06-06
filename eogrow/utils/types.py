""" Includes custom types used in schemas
"""
import datetime
from enum import Enum
from typing import Literal, Tuple, Union

from eolearn.core import FeatureType

Path = str
S3Path = str
ImportPath = str
TimePeriod = Tuple[datetime.date, datetime.date]

Feature = Tuple[FeatureType, str]
FeatureSpec = Union[Tuple[FeatureType, str], FeatureType]

BoolOrAuto = Union[Literal["auto"], bool]


class ProcessingType(Enum):
    RAY = "ray"
    SINGLE = "single"
    MULTI = "multi"
