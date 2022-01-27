"""
Pipelines for testing
"""
import logging
from typing import List, Tuple

from ..core.config import recursive_config_join
from ..core.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


class TestPipeline(Pipeline):
    """Pipeline that just tests if all managers works correctly. It can be used to check if area manager creates a
    correct grid.
    """

    class Schema(Pipeline.Schema):
        class Config:
            extra = "allow"

    _DEFAULT_CONFIG_PARAMS = {
        "pipeline": "eogrow.pipelines.testing.TestPipeline",
        "eopatch": {"manager": "eogrow.eopatches.EOPatchManager"},
        "logging": {"manager": "eogrow.logging.LoggingManager", "show_logs": True},
    }

    def __init__(self, config: dict):
        config = recursive_config_join(config, self._DEFAULT_CONFIG_PARAMS)
        super().__init__(config)

    def run_procedure(self) -> Tuple[List, List]:
        """Performs basic tests of managers"""
        if self.storage.filesystem.exists("/"):
            LOGGER.info("Project folder %s exists", self.storage.config.project_folder)
        else:
            LOGGER.info("Project folder %s does not exist", self.storage.config.project_folder)

        self.area_manager.get_area_dataframe()
        self.area_manager.get_area_geometry()
        grid = self.area_manager.get_grid()
        grid_size = self.area_manager.get_grid_size()
        LOGGER.info("Grid has %d EOPatches and is split over %d CRS zones", grid_size, len(grid))

        eopatches = self.eopatch_manager.get_eopatch_filenames()
        LOGGER.info("The first EOPatch has a name %s", eopatches[0])

        return [], []
