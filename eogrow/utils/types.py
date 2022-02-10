""" Includes custom types used in schemas
"""
from typing import Union, Tuple
import datetime

from eolearn.core import FeatureType

Path = str
S3Path = str
ImportPath = str
TimePeriod = Tuple[datetime.date, datetime.date]

Feature = Union[Tuple[FeatureType, str], FeatureType]
# Feature type is added for BBOX and TIMESTAMP, using Literal with Enum is weird
