"""
Modules with time-related utilities
"""
import datetime as dt

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%SZ"
LOGGING_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_timestamp_suffix(date_only: bool = False) -> str:
    return dt.datetime.utcnow().strftime(DATETIME_FORMAT if not date_only else DATE_FORMAT)
