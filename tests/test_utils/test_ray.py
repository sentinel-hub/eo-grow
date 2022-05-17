"""
Tests for ray utilities
"""
import pytest
import ray

from eogrow.utils.ray import connect_to_ray

pytestmark = pytest.mark.fast


@pytest.fixture(name="ray_cluster", scope="class")
def ray_cluster_fixture():
    ray.init(log_to_driver=False)
    yield
    ray.shutdown()


class TestWithRayCluster:
    """These are the tests that require a running Ray cluster while tests outside this class must not have it."""

    @pytest.mark.parametrize(
        "use_ray, expected_connection",
        [
            ("auto", True),
            (True, True),
            (False, False),
        ],
    )
    def test_connect_to_ray_with_cluster(self, ray_cluster, use_ray, expected_connection):
        is_connected = connect_to_ray(use_ray)
        assert is_connected is expected_connection


@pytest.mark.parametrize(
    "use_ray, expected_connection",
    [
        ("auto", False),
        (True, ConnectionError),
        (False, False),
    ],
)
def test_connect_to_ray_without_cluster(use_ray, expected_connection):
    if isinstance(expected_connection, bool):
        is_connected = connect_to_ray(use_ray)
        assert is_connected is expected_connection
    else:
        with pytest.raises(expected_connection):
            connect_to_ray(use_ray)
