"""
A module with useful utilities related to batch processing
"""
from typing import List


def read_timestamps_from_orbits(userdata: dict) -> List[str]:
    """Parses batch orbits payload to obtain a list of timestamp strings"""
    if "orbits" in userdata:
        userdata = userdata["orbits"]

    return [orbit["tiles"][0]["date"] for orbit in userdata]


def read_timestamps(userdata: dict) -> List[str]:
    """Parses timestamps from a userdata dictionary"""
    return userdata["timestamps"]
