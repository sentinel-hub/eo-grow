import pytest
import ray

from eogrow.utils.ray import handle_ray_connection


@pytest.fixture(name="_ray_cluster", scope="class")
def _ray_cluster_fixture():
    ray.init(log_to_driver=False)
    yield
    ray.shutdown()


class TestWithRayCluster:
    """These are the tests that require a running Ray cluster while tests outside this class must not have it."""

    @pytest.mark.parametrize(
        ("use_ray", "expected_connection"),
        [
            ("auto", True),
            (True, True),
            (False, False),
        ],
    )
    @pytest.mark.usefixtures("_ray_cluster")
    def test_handle_ray_connection_with_cluster(self, use_ray, expected_connection):
        is_connected = handle_ray_connection(use_ray)
        assert is_connected is expected_connection

        if is_connected:
            assert ray.is_initialized()


@pytest.mark.parametrize(
    ("use_ray", "expected_connection"),
    [
        ("auto", False),
        (True, ConnectionError),
        (False, False),
    ],
)
def test_handle_ray_connection_without_cluster(use_ray, expected_connection):
    if isinstance(expected_connection, bool):
        is_connected = handle_ray_connection(use_ray)
        assert is_connected is expected_connection
    else:
        with pytest.raises(expected_connection):
            handle_ray_connection(use_ray)
