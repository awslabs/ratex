import os
import tempfile
from collections import OrderedDict

import pytest
import torch
import torch.optim as optim

import tvm
import torch_mnm
from torch_mnm.utils.cache import Cache
from torch_mnm.testing import TorchLeNet, fake_image_dataset, train



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
        assert cache.persist_path in entry_dir.parents
        assert os.path.exists(os.path.join(entry_dir, cache.TIMESTAMP_FILE))

        # Test cache hit and evit.
        cache = Cache(temp_dir, capacity=1)
        assert len(cache.keys) == 2
        val = cache.query(key1, loader=lambda v: int(v))
        assert val == 123
        entry_dir = cache.query(key2)
        assert cache.persist_path in entry_dir.parents
        assert len(cache.entries) == 1

        # Test pruning by hacking the timestamp file of key2.
        timestamp_file = os.path.join(entry_dir, cache.TIMESTAMP_FILE)
        with open(timestamp_file, "w") as filep:
            filep.write("0")

        pruned_keys = cache.prune_persist(days=1)
        assert len(pruned_keys) == 1 and pruned_keys[0] == key2
        assert len(cache.keys) == 1 and key1 in cache.keys


def test_compile_cache():
    """
    NOTES: This test creates a temporary new global cache, which will affect other tests result
    if multi tests are being exectuted in parallel. Need to revise in the future if we run tests
    in parallel.
    """
    from torch_mnm.utils.cache import cache

    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()

    with tempfile.TemporaryDirectory(prefix="torch_mnm_test_") as temp_dir:
        Cache.__init__(cache, temp_dir)

        train("xla", model, dataset, optimizer=optim.SGD, batch_size=batch_size, num_epochs=1)
        assert cache.misses == 2 and cache.hits == 0

        train("xla", model, dataset, optimizer=optim.SGD, batch_size=batch_size, num_epochs=1)
        assert cache.misses == 2 and cache.hits == 1


def test_convert_module_to_meta_cache():
    from torch_mnm.utils.cache import cache
    model = TorchLeNet()
    with tempfile.TemporaryDirectory(prefix="torch_mnm_test_") as temp_dir:
        # it cannot be accessed with torch_mnm.jit.script.convert_module_to_meta
        from torch_mnm.jit.script import convert_module_to_meta
        # Hook persistent cache to a temporary one
        Cache.__init__(cache, temp_dir)
        args = [torch.rand(1, 1, 28, 28, dtype=torch.float32)]
        shape_n_dtype = (list(args[0].shape), str(args[0].dtype).rsplit(".", maxsplit=1)[-1])
        module = TorchLeNet()
        func, param_names, inplace_update_map, mnm_params_shape, mnm_params_dtype = \
            convert_module_to_meta(module, shape_n_dtype, args)
        assert cache.misses == 2 and cache.hits == 0
        func_1, param_names_1, inplace_update_map_1, mnm_params_shape_1, mnm_params_dtype_1 = \
            convert_module_to_meta(module, shape_n_dtype, args)
        assert cache.misses == 2 and cache.hits == 1
        # clear in-memory cache
        cache.entries = OrderedDict()
        func_1, param_names_1, inplace_update_map_1, mnm_params_shape_1, mnm_params_dtype_1 = \
            convert_module_to_meta(module, shape_n_dtype, args)
        assert isinstance(func, tvm.relay.Function)
        assert tvm.ir.structural_equal(func, func_1)
        assert param_names == param_names_1
        assert inplace_update_map == inplace_update_map_1
        assert mnm_params_shape == mnm_params_shape_1
        assert mnm_params_dtype == mnm_params_dtype_1


if __name__ == "__main__":
    pytest.main([__file__])
