import functools

import boto3
import pytest
from fs.tempfs import TempFS
from fs_s3fs import S3FS
from moto import mock_s3

from eolearn.core import EOExecutor, EONode, EOTask, EOWorkflow

from eogrow.core.logging import EOExecutionFilter, EOExecutionHandler, RegularBackupHandler


class CheckValueTask(EOTask):
    def execute(self, *_, value=0):
        if value != 0:
            raise ValueError(f"Value is {value}")


@mock_s3
def _create_new_s3_fs():
    """Creates a new empty mocked s3 bucket. If one such bucket already exists it deletes it first."""
    bucket_name = "mocked-test-bucket"
    s3resource = boto3.resource("s3", region_name="eu-central-1")

    bucket = s3resource.Bucket(bucket_name)

    if bucket.creation_date:  # If bucket already exists
        for key in bucket.objects.all():
            key.delete()
        bucket.delete()

    s3resource.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})

    return S3FS(bucket_name=bucket_name)


@mock_s3
@pytest.mark.parametrize("fs_loader", [TempFS, _create_new_s3_fs])
@pytest.mark.parametrize(
    "logs_handler_factory", [EOExecutionHandler, functools.partial(RegularBackupHandler, backup_interval=0.01)]
)
@pytest.mark.parametrize(("workers", "multiprocess"), [(1, False), (3, False), (3, True)])
def test_logging_with_eoexecutor(fs_loader, logs_handler_factory, workers, multiprocess):
    """Run EOExecutor and check if reporting and logging was successful"""
    if fs_loader is _create_new_s3_fs and multiprocess:
        # moto mocking doesn't work with multiprocessing
        return

    node1 = EONode(CheckValueTask())
    node2 = EONode(CheckValueTask(), [node1])
    eoworkflow = EOWorkflow([node1, node2])
    execution_args = [{}, {node1: {"value": 1}}, {node2: {"value": 2}}]

    with fs_loader() as temp_fs:
        executor = EOExecutor(
            eoworkflow,
            execution_args,
            save_logs=True,
            logs_folder="logs-folder",
            filesystem=temp_fs,
            logs_handler_factory=logs_handler_factory,
            logs_filter=EOExecutionFilter(ignore_packages=["..."]),
        )
        executor.run(workers=workers, multiprocess=multiprocess)
        executor.make_report(include_logs=False)

        assert temp_fs.exists(executor.get_report_path(full_path=False))
        for log_path in executor.get_log_paths(full_path=False):
            assert temp_fs.exists(log_path)
