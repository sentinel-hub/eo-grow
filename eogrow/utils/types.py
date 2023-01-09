"""Deprecated module for types, moved to `eogrow.types`."""
from warnings import warn

from ..types import *  # noqa # pylint: disable=wildcard-import,unused-wildcard-import

warn(
    "The module `eogrow.utils.types` is deprecated, use `eogrow.types` instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
