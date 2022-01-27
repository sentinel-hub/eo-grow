"""
A module containing general utilities that haven't been sorted in any other module
"""
import datetime as dt
from enum import Enum

from aenum import MultiValueEnum

from sentinelhub import DataCollection
from sentinelhub.data_collections import DataCollectionDefinition


def jsonify(param: object) -> str:
    """Transforms an object into a normal string."""
    if isinstance(param, (dt.datetime, dt.date)):
        return param.isoformat()

    if isinstance(param, DataCollectionDefinition):
        return DataCollection(param).name
    if isinstance(param, DataCollection):
        return param.name

    if isinstance(param, MultiValueEnum):
        return param.values[0]

    if isinstance(param, Enum):
        return param.value

    raise TypeError(f"Object of type {type(param)} is not yet supported in jsonify utility function")
