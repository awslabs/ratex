"""Utilitis of RAZOR Persistent Cache.
TODO: Implement the cache in C++ in the future if we need to cache something in C++.
"""
# pylint: disable=unspecified-encoding
from collections import OrderedDict
import json
import logging
import hashlib
import os
import shutil
import threading
import time

import tvm

logger = logging.getLogger("Cache")

# The path of the persistent cache.
PERSIST_DIR = os.environ.get(
    "RAZOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".torch_mnm_cache")
)


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

    class ThreadSafeWrapper:
        """A thread-safe wrapper of a lock."""

        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    def __init__(self, persist_dir, capacity=None):
        self.persist_dir = persist_dir
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)

        self.enable = self.persist_dir != ""
        self.thread_safe_wrapper = self.ThreadSafeWrapper()

        self.capacity = capacity if capacity is not None else float("inf")
        self.entries = OrderedDict()

        self.keys = self.load_cache_keys() if self.enable else {}

    def normalize_key(self, key):
        """ Normalize the cache key to ensure it's hashable """
        if isinstance(key, list):
            return tuple([self.normalize_key(x) for x in key])
        if isinstance(key, dict):
            return {self.normalize_key(k): self.normalize_key(v) for k, v in key.items()}
        return key

    def load_cache_keys(self):
        """Load the cache keys for fast query."""
        key_file = os.path.join(self.persist_dir, self.KEY_FILE)
        ret = {}
        if os.path.exists(key_file):
            with open(key_file, "r") as filep:
                for entry in json.load(filep):
                    key = self.normalize_key(entry["key"])
                    ret[key] = entry["token"]
        return ret

    def save_cache_keys(self):
        """Save the cache keys to the persistent cache."""
        with open(os.path.join(self.persist_dir, self.KEY_FILE), "w") as filep:
            json.dump([{"key": key, "token": token} for key, token in self.keys.items()], filep)

    def evict(self):
        """Evict cache entries if exceeding the capacity."""
        while len(self.entries) > self.capacity:
            logger.debug("Evit an item from cache")
            self.entries.popitem(last=False)

    @staticmethod
    def get_persist_token(key):
        """Hash the key to be a persist token."""
        return hashlib.md5(str(key).encode(encoding="UTF-8")).hexdigest()

    def get_persist_path(self, token, check_exist=True):
        """Get the path of a persistent cache entry."""
        entry_dir = os.path.join(self.persist_dir, token)
        if check_exist and not os.path.exists(entry_dir):
            raise ValueError(f"Cache entry path is not found: {entry_dir}")
        return entry_dir

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

        # Miss in the persistent cache.
        if key not in self.keys:
            logger.debug("Cache miss in persistent cache: %s", str(key))
            return None

        with self.thread_safe_wrapper:
            # Cache hit.
            if key in self.entries:
                logger.debug("Cache hit: %s", str(key))
                self.entries.move_to_end(key)
                return self.entries[key]

            # Cache miss. Bring from the persistent cache.
            entry_path = self.get_persist_path(self.get_persist_token(key))
            entry_file = os.path.join(entry_path, self.DEFAULT_VALUE_FILE)
            logger.debug("Bring from persistent cache: %s", str(key))

            if os.path.exists(entry_file):
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

        entry_dir = self.create_entry(key)
        entry_file = os.path.join(entry_dir, self.DEFAULT_VALUE_FILE)

        with self.thread_safe_wrapper:
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
        entry_dir: str
            The entry folder path.
        """
        if not self.enable:
            return ""

        with self.thread_safe_wrapper:
            logger.debug("Create a key entry for %s", str(key))
            token = self.get_persist_token(key)
            entry_dir = self.get_persist_path(token, check_exist=False)

            # Create an empty directory.
            if os.path.exists(entry_dir):
                shutil.rmtree(entry_dir)
            os.makedirs(entry_dir)

            # Set the value to be the entry directory path.
            self.entries[key] = entry_dir

            # Write the current timestamp.
            with open(os.path.join(entry_dir, self.TIMESTAMP_FILE), "w") as filep:
                filep.write(str(time.time()))

            # Update and persist the key table.
            if key not in self.keys:
                logger.debug("Update key %s to persistent cache", str(key))
                self.keys[key] = token
                self.save_cache_keys()
            self.evict()
            return entry_dir

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

        with self.thread_safe_wrapper:
            logger.debug("Prune persistent cache entries that are older than %d days", days)

            # Clean all in-memory cache entries.
            self.entries = []

            for key, token in self.keys.items():
                entry_dir = self.get_persist_path(token)
                timestamp_file = os.path.join(entry_dir, self.TIMESTAMP_FILE)
                if not os.path.exists(timestamp_file):
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


cache = Cache(PERSIST_DIR)


def normalize(key):
    if isinstance(key, tvm.ir.container.Array):
        return tuple([normalize(x) for x in key])
    if isinstance(key, tvm.ir.container.Map):
        ks = sorted(key, key=lambda x: normalize(x))
        return tuple([
            (normalize(k), normalize(key[k]))
            for k in ks
        ])
    if isinstance(key, tvm.tir.expr.ConstExpr):
        return key.value
    return hashlib.md5(str(key).encode(encoding="UTF-8")).hexdigest()


@tvm._ffi.register_func("torch_mnm.utils.cache.query")
def query(key):
    ret = cache.query(normalize(key))
    return ret if ret is not None else ""


@tvm._ffi.register_func("torch_mnm.utils.cache.create_entry")
def create_entry(key):
    return cache.create_entry(normalize(key))
