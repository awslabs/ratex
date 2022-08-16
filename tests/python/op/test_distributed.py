# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn
import raf
import ratex
from raf import distributed as dist
from raf.testing import get_dist_comm_info, skip_dist_test
from ratex.lazy_tensor_core.core.lazy_model import lazy_device
from ratex.core.lazy_model import all_gather, all_reduce, reduce_scatter
from ratex.testing import (
    check,
    with_enable_param_aliasing,
)

SKIP_REASON = "Distribution is not enabled or #rank is not expected"


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_all_reduce(dtype):
    """Test of tracing and lowering allreduce op."""
    total_rank, rank, local_rank = get_dist_comm_info()
    n_ones = np.ones(shape=(4, 4), dtype=dtype)
    x = torch.from_numpy(n_ones) * (rank + 1)
    x = x.to(lazy_device(rank))
    y = all_reduce("sum", x, scale=1.0 / total_rank)
    target_y = n_ones * sum(range(1, total_rank + 1)) / total_rank
    check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=4), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("groups", [[[0, 1], [2, 3]], [[0, 1, 2]]])
def test_allreduce_with_subcomm(dtype, groups):
    """Testing allreduce with replica groups."""
    _, rank, local_rank = get_dist_comm_info()
    device = lazy_device(rank)
    x = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
    x = torch.from_numpy(x).to(device)
    y = all_reduce("sum", x, groups=groups)

    is_rank_in_group = False
    for group in groups:
        if rank in group:
            is_rank_in_group = True
            ones = np.ones(shape=(4, 4), dtype=dtype)
            target_y = ones * sum(np.array(group) + 1)
            check(y, target_y)
    if not is_rank_in_group:
        target_y = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_all_gather(dtype):
    """Test of tracing and lowering allgather op."""
    total_rank, rank, local_rank = get_dist_comm_info()
    n_ones = np.ones(shape=(4, 4), dtype=dtype)
    x = torch.from_numpy(n_ones) * (rank + 1)
    x = x.to(lazy_device(rank))
    y = all_gather(x, dim=0)
    target_y = np.concatenate([n_ones * (r + 1) for r in range(total_rank)], axis=0)
    check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=4), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("groups", [[[0, 1], [2, 3]], [[0, 1, 2]]])
def test_allgather_with_subcomm(dtype, groups):
    """Testing allgather with with replica groups."""
    _, rank, local_rank = get_dist_comm_info(verbose=True)
    device = lazy_device(rank)
    n_x = np.ones(shape=(4, 4), dtype=dtype)
    x = torch.from_numpy(n_x) * (rank + 1)
    x = x.to(device)
    y = all_gather(x, dim=0, groups=groups)

    is_rank_in_group = False
    for group in groups:
        if rank in group:
            target_y = np.concatenate([n_x * (r + 1) for r in group])
            is_rank_in_group = True
            check(y, target_y)

    if not is_rank_in_group:
        target_y = n_x * (rank + 1)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@with_enable_param_aliasing
def test_all_gather_out(dtype):
    """Test of tracing and lowering allgather op."""
    total_rank, rank, local_rank = get_dist_comm_info()
    shape = [1, 1, 28, 28]
    expected_ret_shape = shape.copy()
    expected_ret_shape[0] *= 4
    device = lazy_device(rank)
    n_ones = np.ones(shape=tuple(shape), dtype=dtype)
    x = torch.from_numpy(n_ones) * (rank + 1)
    x = x.to(device)
    y = torch.zeros(*expected_ret_shape).to(device)
    all_gather(x, dim=0, output=y)
    target_y = np.concatenate([n_ones * (r + 1) for r in range(total_rank)], axis=0)
    check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_reduce_scatter(dtype):
    """Test of tracing and lowering reduce_scatter op."""
    total_rank, rank, local_rank = get_dist_comm_info()
    device = lazy_device(rank)
    n_ones = np.ones(shape=(4, 4), dtype=dtype)
    tmps = []
    for i in range(total_rank):
        tmps.append(n_ones * (rank + i))
    x = torch.from_numpy(np.concatenate(tmps))
    x = x.to(device)
    ret = reduce_scatter(x, "sum")
    target_ret = n_ones * sum(range(rank, total_rank + rank))
    check(ret, target_ret)


@pytest.mark.skipif(skip_dist_test(min_rank_num=4, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "min", "max"])
@pytest.mark.parametrize("rank_list", [[[0, 1], [2, 3]]])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_reduce_scatter_with_rank_list(computation, rank_list, dtype):
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = lazy_device(rank)
    n_ones = np.ones(shape=(4, 4), dtype=dtype)
    x1 = torch.from_numpy(n_ones * (rank + 1))
    x2 = torch.from_numpy(-n_ones * (rank + 1))
    inputs = torch.cat([x1, x2]).to(device)
    ret = reduce_scatter(inputs, computation, groups=rank_list)

    for group in rank_list:
        if rank in group:
            ones = np.ones(shape=(4, 4), dtype=dtype)
            if computation == "sum":
                even_out = ones * sum(np.array(group) + 1)
                odd_out = -even_out
            elif computation == "prod":
                even_out = ones * np.prod([(temp_rank + 1) for temp_rank in np.array(group)])
                odd_out = even_out
            elif computation == "min":
                even_out = ones * min(np.array(group) + 1)
                odd_out = -ones * max(np.array(group) + 1)
            elif computation == "max":
                even_out = ones * max(np.array(group) + 1)
                odd_out = -ones * min(np.array(group) + 1)
            elif computation == "avg":
                even_out = ones * sum(np.array(group) + 1)
                even_out = even_out / total_rank
                odd_out = -even_out
            if rank % 2 == 0:
                check(ret, even_out)
            else:
                check(ret, odd_out)


if __name__ == "__main__":
    if os.environ.get("RAF_FILE_STORE_PATH", None):
        dist.set_default_communicator("void")
        comm = dist.get_communicator()
        size = int(
            os.environ.get("OMPI_COMM_WORLD_SIZE", None) or os.environ.get("MPIRUN_NPROCS", None)
        )
        rank = int(
            os.environ.get("OMPI_COMM_WORLD_RANK", None) or os.environ.get("MPIRUN_RANK", None)
        )
        comm.size = size
        comm.rank = rank
        comm.local_size = size
        comm.local_rank = rank
    exit_code = pytest.main([__file__, "-s"])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
