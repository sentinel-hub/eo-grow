"""
A module with useful utilities related to batch processing
"""

from __future__ import annotations


def read_timestamps_from_orbits(userdata: dict) -> list[str]:
    """Parses batch orbits payload to obtain a list of timestamp strings"""
    if "orbits" in userdata:  # SIM908
        userdata = userdata["orbits"]

    return [orbit["tiles"][0]["date"] for orbit in userdata]


def read_timestamps(userdata: dict) -> list[str]:
    """Parses timestamps from a userdata dictionary"""
    return userdata["timestamps"]
