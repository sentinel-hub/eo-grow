"""
Module containing utilities for working with filesystems
"""
from typing import Optional

import fs
import fs.copy
from fs.osfs import OSFS
from fs.tempfs import TempFS

from sentinelhub import SHConfig
from eolearn.core.utils.fs import get_base_filesystem_and_path


class LocalFile:
    """An abstraction for working with a local version of a file.

    If file's original location is remote (e.g. on S3) then this will ensure working with a local copy. If file's
    original location is already on local filesystem then it will work with that, unless `always_copy=True` is used.
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        filesystem: Optional[fs.base.FS] = None,
        config: Optional[SHConfig] = None,
        always_copy: bool = False,
        **temp_fs_kwargs,
    ):
        """
        :param path: Either a full path to a remote file or a path to remote file which is relative to given filesystem
            object.
        :type path: str
        :param mode: One of the option `r', 'w', and 'rw`, which specify if a file should be read or written to remote.
            The default is 'r'.
        :type mode: str
        :param filesystem: A filesystem of the remote. If not given, it will be determined from the path.
        :type: fs.FS
        :param config: A config object with which AWS credentials could be used to initialize a remote filesystem object
        :type config: SHConfig
        :param always_copy: If True it will always make a local copy to a temporary folder, even if a file is already
            in the local filesystem.
        :type always_copy: bool
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
            self._filesystem = TempFS(**temp_fs_kwargs)
            self._local_path = fs.path.basename(self._remote_path)

        self._absolute_local_path = self._filesystem.getsyspath(self._local_path)

        if "r" in self._mode:
            self.copy_to_local()
        elif "w" in self._mode:
            remote_dirs = fs.path.dirname(self._remote_path)
            self._remote_filesystem.makedirs(remote_dirs, recreate=True)

    @property
    def path(self) -> str:
        """Provides an absolute path to the copy in the local filesystem"""
        return self._absolute_local_path

    def __enter__(self):
        """This allows the class to be used as a context manager"""
        return self

    def __exit__(self, *_, **__):
        """This allows the class to be used as a context manager. In case an error is raised this will by default
        still delete the object from local folder. In case you don't want that, initialize `LocalFile` with
        `auto_clean=False`.
        """
        self.close()

    def close(self):
        """Close the local copy"""
        if "w" in self._mode:
            self.copy_to_remote()

        if self._filesystem is not self._remote_filesystem:
            self._filesystem.close()

    def copy_to_local(self):
        """Copy from remote to local location"""
        if self._filesystem is not self._remote_filesystem:
            fs.copy.copy_file(self._remote_filesystem, self._remote_path, self._filesystem, self._local_path)

    def copy_to_remote(self):
        """Copy from local to remote location"""
        if self._filesystem is not self._remote_filesystem:
            fs.copy.copy_file(self._filesystem, self._local_path, self._remote_filesystem, self._remote_path)
