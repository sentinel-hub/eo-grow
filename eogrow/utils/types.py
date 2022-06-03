""" Includes custom types used in schemas
"""
import datetime
from typing import Literal, Tuple, Union

from eolearn.core import FeatureType

Path = str
S3Path = str
ImportPath = str
TimePeriod = Tuple[datetime.date, datetime.date]

Feature = Tuple[FeatureType, str]
FeatureSpec = Union[Tuple[FeatureType, str], FeatureType]

BoolOrAuto = Union[Literal["auto"], bool]

ExecutionKind = Literal["single", "multi", "ray"]
