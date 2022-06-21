# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.optim as optim

import ratex
import tvm
from ratex.testing import TorchLeNet, fake_image_dataset, train
from ratex.testing import with_dumped_tensor_file, with_temp_cache
from ratex.utils.cache import Cache


def test_cache():
    with tempfile.TemporaryDirectory(prefix="ratex_test_") as temp_dir:
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


@with_temp_cache
def test_compile_cache():
    from ratex.jit.script import JIT_CACHE
    from ratex.utils.cache import cache

    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()

    JIT_CACHE.clear()

    train("lazy", model, dataset, optimizer=optim.SGD, batch_size=batch_size, num_epochs=1)
    assert cache.misses == 1 and cache.hits == 0

    # Clear the JIT cache to force compile
    JIT_CACHE.clear()

    train("lazy", model, dataset, optimizer=optim.SGD, batch_size=batch_size, num_epochs=1)
    assert cache.misses == 1 and cache.hits == 1


@with_temp_cache
def test_convert_module_to_meta_cache():
    # it cannot be accessed with ratex.jit.script.convert_module_to_raf
    from ratex.jit.script import convert_module_to_raf
    from ratex.utils.cache import cache

    args = [torch.rand(1, 1, 28, 28, dtype=torch.float32)]
    shape_n_dtype = (list(args[0].shape), str(args[0].dtype).rsplit(".", maxsplit=1)[-1])
    module = TorchLeNet()
    (
        func,
        param_names,
        inplace_update_map,
        raf_params_shape,
        raf_params_dtype,
    ) = convert_module_to_raf(module, shape_n_dtype, args)
    assert cache.misses == 1 and cache.hits == 0
    (
        func_1,
        param_names_1,
        inplace_update_map_1,
        raf_params_shape_1,
        raf_params_dtype_1,
    ) = convert_module_to_raf(module, shape_n_dtype, args)
    assert cache.misses == 1 and cache.hits == 1
    # clear in-memory cache
    cache.evict_all()
    (
        func_1,
        param_names_1,
        inplace_update_map_1,
        raf_params_shape_1,
        raf_params_dtype_1,
    ) = convert_module_to_raf(module, shape_n_dtype, args)
    assert isinstance(func, tvm.relay.Function)
    assert tvm.ir.structural_equal(func, func_1)
    assert param_names == param_names_1
    assert inplace_update_map == inplace_update_map_1
    assert raf_params_shape == raf_params_shape_1
    assert raf_params_dtype == raf_params_dtype_1


@with_temp_cache
@with_dumped_tensor_file
def test_convert_module_to_meta_cache_ltc_trace():
    """This is to test if the LTC IR is the same before and after JIT compile hits."""
    from ratex.jit.script import JIT_CACHE
    from ratex.utils.cache import cache

    batch_size = 16
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()

    train(
        "lazy",
        model,
        dataset,
        optimizer=optim.SGD,
        batch_size=batch_size,
        num_epochs=1,
        trim=True,
    )

    ltc_file = os.environ["LTC_SAVE_TENSORS_FILE"]
    with open(ltc_file, "r") as file:
        hashes_before_hit = [l for l in file if re.search("Hashes", l)]
        # Last two are hashes for the model and the optimizer
        hashes_before_hit = hashes_before_hit[-2:]

    # Clear in-memory cache
    cache.evict_all()
    JIT_CACHE.clear()

    train(
        "lazy",
        model,
        dataset,
        optimizer=optim.SGD,
        batch_size=batch_size,
        num_epochs=1,
        trim=True,
    )

    with open(ltc_file, "r") as file:
        hashes_after_hit = [l for l in file if re.search("Hashes", l)]
        # Last two are hashes for the model and the optimizer
        hashes_after_hit = hashes_after_hit[-2:]

    assert hashes_before_hit == hashes_after_hit


if __name__ == "__main__":
    pytest.main([__file__])
