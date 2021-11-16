import os
import tempfile

import pytest

from torch_mnm.utils.cache import Cache

def test_cache():
    with tempfile.TemporaryDirectory(prefix="torch_mnm_test_") as temp_dir:
        # Test cache miss and commit.
        cache = Cache(temp_dir)

        key1 = ("str key", 2)
        val = cache.query(key1)
        assert val is None

        val = cache.commit(key1, 123, saver=lambda v: str(v))
        assert val == 123
        assert key1 in cache.keys
        assert val == cache.entries[key1]        

        token = cache.keys[key1]
        entry_dir = os.path.join(temp_dir, token)
        assert os.path.exists(os.path.join(entry_dir, cache.DEFAULT_VALUE_FILE))
        assert os.path.exists(os.path.join(entry_dir, cache.TIMESTAMP_FILE))

        key2 = "aaa"
        entry_dir = cache.create_entry(key2)
        assert entry_dir.startswith(temp_dir)
        assert os.path.exists(os.path.join(entry_dir, cache.TIMESTAMP_FILE))

        # Test cache hit and evit.
        cache = Cache(temp_dir, capacity=1)
        assert len(cache.keys) == 2
        val = cache.query(key1, loader=lambda v: int(v))
        assert val == 123
        entry_dir = cache.query(key2)
        assert entry_dir.startswith(temp_dir)
        assert len(cache.entries) == 1

        # Test pruning by hacking the timestamp file of key2.
        timestamp_file = os.path.join(entry_dir, cache.TIMESTAMP_FILE)
        with open(timestamp_file, "w") as filep:
            filep.write("0")

        pruned_keys = cache.prune_persist(days=1)
        assert len(pruned_keys) == 1 and pruned_keys[0] == key2
        assert len(cache.keys) == 1 and key1 in cache.keys


if __name__ == "__main__":
    pytest.main([__file__])
