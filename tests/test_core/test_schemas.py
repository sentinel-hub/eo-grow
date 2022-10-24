"""
Tests for schemas module
"""
import pytest

from eogrow.core.area import UtmZoneAreaManager
from eogrow.core.schemas import build_minimal_template, build_schema_template
from eogrow.core.storage import StorageManager
from eogrow.pipelines.export_maps import ExportMapsPipeline
from eogrow.pipelines.mapping import MappingPipeline

pytestmark = pytest.mark.fast


@pytest.mark.fast
@pytest.mark.parametrize("eogrow_object", [UtmZoneAreaManager, MappingPipeline, ExportMapsPipeline, StorageManager])
@pytest.mark.parametrize("schema_builder", [build_schema_template, build_minimal_template])
def test_build_schema_template(eogrow_object, schema_builder):
    template = schema_builder(eogrow_object.Schema)
    assert isinstance(template, dict)
