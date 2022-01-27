"""
Tests for schemas module
"""
import pytest

from eogrow.core.area import UtmZoneAreaManager
from eogrow.core.schemas import build_schema_template
from eogrow.core.storage import StorageManager
from eogrow.pipelines.export_maps import ExportMapsPipeline
from eogrow.pipelines.download import DownloadPipeline


@pytest.mark.fast
@pytest.mark.parametrize("eogrow_object", [UtmZoneAreaManager, DownloadPipeline, ExportMapsPipeline, StorageManager])
def test_build_schema_template(eogrow_object):
    template = build_schema_template(eogrow_object.Schema)
    assert isinstance(template, dict)
