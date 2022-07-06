# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilitis of RATEX Persistent Cache.
TODO: Implement the cache in C++ in the future if we need to cache something in C++.
"""
# pylint: disable=protected-access, too-many-instance-attributes, abstract-class-instantiated
from collections import OrderedDict
from pathlib import Path
import json
import logging
import hashlib
import os
import shutil
import time
from filelock import FileLock

import tvm

logger = logging.getLogger("Cache")  # pylint: disable=invalid-name

# The path of the persistent cache.
PERSIST_DIR = os.environ.get("RATEX_CACHE_DIR", Path.home() / ".ratex_cache")


class Cache:
    """An in-memory LRU cache with a persistent storage in disk. Note that the persistent
    storage will not be evited/cleaned automatically. Users need to explicitly call
    `prune_persist()` to free the disk space.

    Parameters
    ----------
    persist_dir: str
        The path of the persistent cache. If the path does not exist, it will be created.
        If empty, the cache is disabled.

    capacity: int
        The capacity of the cache. If the number of in-memory entries exceeds the capacity,
        the least recently used entries will be evicted. If None, the cache is unbounded.
    """

    KEY_FILE = "keys.json"
    DEFAULT_VALUE_FILE = "_cache_value_file"
    TIMESTAMP_FILE = "timestamp"

    def __init__(self, persist_dir, capacity=None):
        self.persist_path = ""
        self.enable = False

        if persist_dir != "":
            self.persist_path = Path(persist_dir)
            self.enable = True
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self.file_lock = FileLock(self.persist_path / (self.KEY_FILE + ".lock"))

        self.capacity = capacity if capacity is not None else float("inf")
        self.entries = OrderedDict()
        self.entry_locks = {}

        self.keys = self.load_cache_keys() if self.enable else {}
        self.hits = 0
        self.misses = 0

    def normalize_key(self, key):
        """Normalize the cache key to ensure it's hashable"""
        if isinstance(key, list):
            return tuple(self.normalize_key(x) for x in key)
        if isinstance(key, dict):
            return {self.normalize_key(k): self.normalize_key(v) for k, v in key.items()}
        return key

    def load_cache_keys(self):
        """Load the cache keys for fast query."""
        key_file = self.persist_path / self.KEY_FILE
        ret = {}
        if key_file.exists():
            with open(key_file, "r") as filep:
                for entry in json.load(filep):
                    key = self.normalize_key(entry["key"])
                    ret[key] = entry["token"]
        return ret

    def save_cache_keys(self):
        """Save the cache keys to the persistent cache."""
        with open(self.persist_path / self.KEY_FILE, "w") as filep:
            json.dump([{"key": key, "token": token} for key, token in self.keys.items()], filep)

    def evict(self):
        """Evict cache entries if exceeding the capacity."""
        while len(self.entries) > self.capacity:
            logger.debug("Evit an item from cache")
            self.entries.popitem(last=False)

    def evict_all(self):
        """Evict all entries"""
        self.entries = OrderedDict()
        self.entry_locks = {}

    @staticmethod
    def get_persist_token(key):
        """Hash the key to be a persist token."""
        return hashlib.md5(str(key).encode(encoding="UTF-8")).hexdigest()

    def get_persist_path(self, token, check_exist=True):
        """Get the path of a persistent cache entry."""
        entry_path = self.persist_path / token
        if check_exist and not entry_path.exists():
            raise ValueError(f"Cache entry path is not found: {entry_path}")
        return entry_path

    def query(self, key, loader=None):
        """Query the cache value.

        Parameters
        ----------
        key: Hashable
            The hashable cache key.

        loader: Callable
            The loader function to load the cache value from file.

        Returns
        -------
        value: Optional[Any]
            The value of the cache, or None if key is not found.
        """
        if not self.enable:
            return None

        with self.file_lock:
            # Load the cache keys file to sync with other processes
            self.keys.update(self.load_cache_keys())

            # Miss in the persistent cache.
            if key not in self.keys:
                logger.debug("Cache miss in persistent cache: %s", str(key))
                self.misses += 1
                return None

            # Cache hit.
            if key in self.entries:
                logger.debug("Cache hit: %s", str(key))
                self.entries.move_to_end(key)
                self.hits += 1
                return self.entries[key]

            # Cache miss. Bring from the persistent cache.
            token = self.get_persist_token(key)
            entry_path = self.get_persist_path(token)
            entry_file = entry_path / self.DEFAULT_VALUE_FILE
            logger.debug("Bring from persistent cache: %s, token:%s", str(key), str(token))

        if entry_file.exists():
            # If the default cache value file exists, we assume the value was written
            # by the commit() function, and we directly load the value from the file
            # to speedup future queries.
            with open(entry_file, "r") as filep:
                value = filep.read()
                if loader is not None:
                    value = loader(value)
        else:
            # Otherwise, just load the file path and let users access the file directly.
            if loader is not None:
                raise RuntimeError(
                    f"Loader is not applicable to the user-managed cache entries: {str(key)}"
                )
            value = entry_path

        with self.file_lock:
            if key in self.entries and self.entries[key] != value:
                raise RuntimeError(f"Thread data racing: {str(key)}")
            self.entries[key] = value
            self.evict()

        return value

    def commit(self, key, value, saver=None):
        """Commit a new entry to cache. If the key is already in the cache, the value will be
        overwritten.

        Parameters
        ----------
        key: Hashable
            The hashable cache key.

        value: Any
            The value of the cache.

        saver: Optional[Callable]
            The saver function to serialize the value so that it can be saved to file.

        Returns
        -------
        value: Any
            The cached value.
        """
        if not self.enable:
            return value

        entry_path = self.create_entry(key)
        entry_file = entry_path / self.DEFAULT_VALUE_FILE

        with self.file_lock:
            logger.debug("Commit %s to persistent cache", str(key))

            # Write the value to a file.
            with open(entry_file, "w") as filep:
                filep.write(saver(value) if saver is not None else value)

            # Update the value from the entry file path to the in-memory data.
            self.entries[key] = value
            return value

    def create_entry(self, key):
        """Create a new entry by given the key. The value will be the entry file path in the
        persistent storage. Note that if the key already exists, then its value will be removed.

        Parameters
        ----------
        key: Hashable
            The hashable cache key.

        Returns
        -------
        entry_path: Optional[Path]
            The entry folder pathm, or None if cache is disabled.
        """
        if not self.enable:
            return None

        with self.file_lock:
            logger.debug("Create a key entry for %s", str(key))
            token = self.get_persist_token(key)
            entry_path = self.get_persist_path(token, check_exist=False)

            # Create an empty directory.
            if entry_path.exists():
                shutil.rmtree(entry_path)
            entry_path.mkdir(parents=True)

            # Set the value to be the entry directory path.
            self.entries[key] = entry_path

            # Write the current timestamp.
            with open(entry_path / self.TIMESTAMP_FILE, "w") as filep:
                filep.write(str(time.time()))

            # Update and persist the key table.
            if key not in self.keys:
                logger.debug("Update key %s to persistent cache, token %s", str(key), str(token))
                self.keys[key] = token
                # Load the cache keys file to sync with other processes
                self.keys.update(self.load_cache_keys())
                self.save_cache_keys()
            self.evict()
            return entry_path

    def prune_persist(self, days):
        """Prune the persistent cache entries that are older than the given number of days.
        Note that this will clean all current in-memory cache entries so this function should
        be used longly.

        Parameters
        ----------
        days: int
            The number of days to be preserved.

        Returns
        -------
        pruned_keys: List[Hashable]
            The list of keys that are pruned.
        """
        if not self.enable:
            logger.warning("Cache is not enabled for pruning")
            return []

        prunted_keys = []

        with self.file_lock:
            logger.debug("Prune persistent cache entries that are older than %d days", days)

            # Clean all in-memory cache entries.
            self.evict_all()

            for key, token in self.keys.items():
                entry_dir = self.get_persist_path(token)
                timestamp_file = entry_dir / self.TIMESTAMP_FILE
                if not timestamp_file.exists():
                    logger.warning(
                        "Cannot determine whether %s can be pruned: Timestamp not found", str(key)
                    )

                with open(timestamp_file, "r") as filep:
                    timestamp_str = filep.read()
                    try:
                        timestamp = float(timestamp_str)
                    except ValueError:
                        logger.warning(
                            "Cannot determine whether %s can be pruned: Invalid timestamp %s",
                            key,
                            timestamp_str,
                        )
                        continue

                if time.time() - timestamp > days * 24 * 3600:
                    logger.debug("Remove key %s from persistent cache", str(key))
                    prunted_keys.append(key)
                    shutil.rmtree(entry_dir)

            for key in prunted_keys:
                del self.keys[key]

            # Update the pruned key table.
            self.save_cache_keys()
        return prunted_keys

    def acquire_cache_entry_lock(self, key):
        """Aquire a lock to an entry by given the key.

        Parameters
        ----------
        key: Hashable
            The hashable cache key.
        """
        token = self.get_persist_token(key)
        if key not in self.entry_locks:
            self.entry_locks[key] = FileLock(self.persist_path / (token + ".lock"))
        self.entry_locks[key].acquire()

    def release_cache_entry_lock(self, key):
        """Release a lock to an entry by given the key.

        Parameters
        ----------
        key: Hashable
            The hashable cache key.
        """
        assert key in self.entry_locks
        self.entry_locks[key].release()


cache = Cache(PERSIST_DIR)  # pylint: disable=invalid-name


def normalize(key):
    """Normalize the given key by hashing it with MD5."""
    if isinstance(key, tvm.ir.container.Array):
        return tuple(normalize(x) for x in key)
    if isinstance(key, tvm.ir.container.Map):
        keys = sorted(key, key=normalize)
        return tuple((normalize(k), normalize(key[k])) for k in keys)
    if isinstance(key, tvm.tir.expr.ConstExpr):
        return key.value
    # FIXME: raf.ir.AsText(key) segfaults, because we do not have may_share field
    # if isinstance(key, tvm.relay.Expr):
    #     return hashlib.md5(raf.ir.AsText(key).encode(encoding="UTF-8")).hexdigest()
    return hashlib.md5(str(key).encode(encoding="UTF-8")).hexdigest()


@tvm._ffi.register_func("ratex.utils.cache.query")
def query(key):
    """A helper function to query the cache for the given key in C++."""
    ret = cache.query(normalize(key))
    return str(ret) if ret is not None else ""


@tvm._ffi.register_func("ratex.utils.cache.create_entry")
def create_entry(key):
    """A helper function to create an entry for the given key in C++."""
    return str(cache.create_entry(normalize(key)))


@tvm._ffi.register_func("ratex.utils.cache.get_persist_token")
def get_persist_token(key):
    """A helper function to create a token for the given key in C++."""
    return cache.get_persist_token(normalize(key))


@tvm._ffi.register_func("ratex.utils.cache.acquire_cache_entry_lock")
def acquire_cache_entry_lock(key):
    """Acquire the lock associated with the entry"""
    cache.acquire_cache_entry_lock(normalize(key))


@tvm._ffi.register_func("ratex.utils.cache.release_cache_entry_lock")
def release_cache_entry_lock(key):
    """Release the lock associated with the entry"""
    cache.release_cache_entry_lock(normalize(key))


def copy_cache(src_cache, tgt_cache, days):
    """Copy cache to another directory.

    Parameters
    ----------
    src_cache: string
        The source directory.

    tgt_cache: string
        The target directory.

    days: int
        The number of days to be preserved.
    """
    assert os.path.isdir(src_cache)
    assert not os.path.isdir(tgt_cache)
    shutil.copytree(src_cache, tgt_cache)
    new_cache = Cache(tgt_cache)
    new_cache.prune_persist(days)
