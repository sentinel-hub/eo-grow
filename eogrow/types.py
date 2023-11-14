""" Includes custom types used in schemas
"""

import datetime
import sys
from typing import Any, Dict, List, Tuple

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from eolearn.core import EONode
from sentinelhub import BBox

PatchList: TypeAlias = List[Tuple[str, BBox]]
ExecKwargs: TypeAlias = Dict[str, Dict[EONode, Dict[str, object]]]

ImportPath: TypeAlias = str
TimePeriod: TypeAlias = Tuple[datetime.date, datetime.date]

JsonDict: TypeAlias = Dict[str, Any]
RawSchemaDict: TypeAlias = Dict[str, Any]
