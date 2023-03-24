"""Implements utility tasks for importing vector data from a file."""
import datetime as dt

from eolearn.core import EOPatch, EOTask
from eolearn.core.constants import TIMESTAMP_COLUMN

from ..types import Feature


class ExtractTimestampsTask(EOTask):
    def __init__(self, input_feature: Feature):
        self.input_feature = input_feature

    def execute(self, eopatch: EOPatch) -> EOPatch:
        gdf = eopatch[self.input_feature]
        eopatch.timestamps = sorted(
            dt.datetime.fromisoformat(timestamp) for timestamp in gdf[TIMESTAMP_COLUMN].unique()
        )
        return eopatch
