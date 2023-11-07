"""
Module containing utilities for working with filesystems
"""

from __future__ import annotations

import abc
import os
from typing import Any, TypeVar

import fs
import fs.copy
from fs.base import FS
from fs.errors import ResourceNotFound
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs.walk import Walker

from eolearn.core.utils.fs import get_base_filesystem_and_path
from sentinelhub import SHConfig

Self = TypeVar("Self", bound="BaseLocalObject")


class BaseLocalObject(metaclass=abc.ABCMeta):
    """An abstraction for working with a local version of remote objects.

    If object's original location is remote (e.g. on S3) then this will ensure working with a local copy. If object's
    original location is already on local filesystem then it will work with that, unless `always_copy=True` is used.
    """

    def __init__(
        self,
        path: str,
        *,
        mode: str = "r",
        filesystem: FS | None = None,
        config: SHConfig | None = None,
        always_copy: bool = False,
        **temp_fs_kwargs: Any,
    ):
        """
        :param path: Either a full path to a remote file or a path to remote file which is relative to given filesystem
            object.
        :param mode: One of the option `r', 'w', and 'rw`, which specify if a file should be read or written to remote.
            The default is 'r'.
        :param filesystem: A filesystem of the remote. If not given, it will be determined from the path.
        :param config: A config object with which AWS credentials could be used to initialize a remote filesystem
            object.
        :param always_copy: If True it will always make a local copy to a temporary folder, even if a file is already
            in the local filesystem.
        :param temp_fs_kwargs: Parameters that will be propagated to fs.tempfs.TempFS
        """
        if filesystem is None:
            filesystem, path = get_base_filesystem_and_path(path, config=config)
        self._remote_path = path
        self._remote_filesystem = filesystem

        if not (mode and isinstance(mode, str) and set(mode) <= {"r", "w"}):
            raise ValueError(f"Parameter mode should be one of the strings 'r', 'w' or 'rw' but {mode} found")
        self._mode = mode

        if isinstance(self._remote_filesystem, (OSFS, TempFS)) and not always_copy:
            self._filesystem = self._remote_filesystem
            self._local_path = self._remote_path
        else:
            self._add_tempfs_identifier(temp_fs_kwargs)
            self._filesystem = TempFS(**temp_fs_kwargs)
            self._local_path = fs.path.basename(self._remote_path)

        self._absolute_local_path = self._filesystem.getsyspath(self._local_path)
        self._remote_location_ensured = False

        if "r" in self._mode:
            self.copy_to_local()
        if "w" in self._mode and self._filesystem is self._remote_filesystem:
            self._ensure_remote_location()

    @property
    def path(self) -> str:
        """Provides an absolute path to the copy in the local filesystem"""
        return self._absolute_local_path

    def __enter__(self: Self) -> Self:
        """This allows the class to be used as a context manager"""
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        """This allows the class to be used as a context manager. In case an error is raised this will by default
        still delete the object from local folder.
        """
        self.close()

    def close(self) -> None:
        """Close the local copy"""
        if "w" in self._mode:
            self.copy_to_remote(raise_missing=False)

        if self._filesystem is not self._remote_filesystem:
            self._filesystem.close()

    def copy_to_local(self) -> None:
        """A public method for copying from remote to local location. It can be called anytime."""
        if self._filesystem is not self._remote_filesystem:
            self._copy_to_local()

    def copy_to_remote(self, raise_missing: bool = True) -> None:
        """Copy from local to remote location"""
        if not self._filesystem.exists(self._local_path):
            if raise_missing:
                raise ResourceNotFound(f"Local resource {self._local_path} doesn't exist")
            return

        if self._filesystem is not self._remote_filesystem:
            self._ensure_remote_location()
            self._copy_to_remote()

    def _add_tempfs_identifier(self, temp_fs_kwargs: dict[str, Any]) -> None:
        """Adds an identifier name that will be used as a suffix of a temporary local folder. This is helpful for
        debugging purposes."""
        if "identifier" in temp_fs_kwargs:
            return

        object_name = fs.path.basename(self._remote_path.rstrip("/")).split(".", 1)[0]
        temp_fs_kwargs["identifier"] = f"_{self.__class__.__name__}-{object_name}"

    def _ensure_remote_location(self) -> None:
        """Makes sure that the remote location exists. If it doesn't then it will try to create the missing folders.
        This method is also regulated with a flag so that the IO checks happen at most once."""
        if not self._remote_location_ensured:
            remote_dirs = fs.path.dirname(self._remote_path)
            self._remote_filesystem.makedirs(remote_dirs, recreate=True)
            self._remote_location_ensured = True

    @abc.abstractmethod
    def _copy_to_local(self) -> None:
        """Copy from remote to local location"""

    @abc.abstractmethod
    def _copy_to_remote(self) -> None:
        """Copy from local to remote location"""


class LocalFile(BaseLocalObject):
    """An abstraction for working with a local version of a remote file.

    Check `BaseLocalObject` for more info.
    """

    def _copy_to_local(self) -> None:
        """Copy the file from remote to local location."""
        fs.copy.copy_file(self._remote_filesystem, self._remote_path, self._filesystem, self._local_path)

    def _copy_to_remote(self) -> None:
        """Copy the file from local to remote location."""
        fs.copy.copy_file(self._filesystem, self._local_path, self._remote_filesystem, self._remote_path)


class LocalFolder(BaseLocalObject):
    """An abstraction for working with a local version of a remote folder and its content.

    Check `BaseLocalObject` for more info.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filesystem: FS | None = None,
        config: SHConfig | None = None,
        walker: Walker | None = None,
        workers: int | None = None,
        **kwargs: Any,
    ):
        """
        :param path: Either a full path to a remote folder or a path to remote folder which is relative to given
            filesystem object.
        :param args: Positional arguments propagated to the base class.
        :param filesystem: A filesystem of the remote. If not given, it will be determined from the path.
        :param config: A config object with which AWS credentials could be used to initialize a remote filesystem
            object.
        :param walker: An instance of `fs.walk.Walker` object used to configure advanced copying parameters. It is
            possible to set max folder depth of copy (default is entire tree), how to handle errors (by default are
            ignored), and what to include or exclude.
        :param workers: A maximal number of threads used for parallel copies between local and remote locations. The
            default is `5` times the number of CPUs.
        :param kwargs: Keyword arguments propagated to the base class.
        """
        self.walker = walker or Walker(ignore_errors=True)
        self.workers = 5 * (os.cpu_count() or 1) if workers is None else workers

        if filesystem is None:
            filesystem, path = get_base_filesystem_and_path(path, config=config)
        if not path.endswith("/"):
            path += "/"  # This way fs will treat the path as a folder

        super().__init__(path, *args, filesystem=filesystem, **kwargs)

    def _copy_to_local(self) -> None:
        """Copy the folder content from remote to local location."""
        fs.copy.copy_dir(
            self._remote_filesystem,
            self._remote_path,
            self._filesystem,
            self._local_path,
            walker=self.walker,
            workers=self.workers,
        )

    def _copy_to_remote(self) -> None:
        """Copy the folder content from local to remote location."""
        fs.copy.copy_dir(
            self._filesystem,
            self._local_path,
            self._remote_filesystem,
            self._remote_path,
            walker=self.walker,
            workers=self.workers,
        )
