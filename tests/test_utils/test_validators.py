"""
Tests for validator utilities
"""
import datetime as dt

import pytest

from eogrow.utils.validators import parse_time_period

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "time_period,year,expected_start_date,expected_end_date",
    [
        ("yearly", 2020, "2020-01-01", "2020-12-31"),
        ("Q2", 2021, "2021-04-01", "2021-06-30"),
        ("Q2-yearly", 2021, "2020-07-01", "2021-06-30"),
    ],
)
def test_parse_time_period(time_period, year, expected_start_date, expected_end_date):
    start_date, end_date = parse_time_period([time_period, year])

    assert isinstance(start_date, dt.date)
    assert isinstance(end_date, dt.date)

    assert start_date.isoformat() == expected_start_date
    assert end_date.isoformat() == expected_end_date
