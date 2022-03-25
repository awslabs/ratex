# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
import raf
import razor
from raf import distributed as dist
from razor.core.lazy_model import all_gather, all_reduce
from razor.testing import compile_model, with_enable_param_aliasing, with_mock_distributed_context


@patch("raf.distributed.get_context")
@pytest.mark.parametrize("world_size", [1, 4])
def test_all_reduce(mock_get_context, world_size):
    """Test of tracing and lowering allreduce op."""

    # Mock the dist context.
    class MockContext:
        def __init__(self):
            self.enable_data_parallel = False
            self.zero_opt_level = 0
            self.size = world_size
            self.rank = 0

    mock_get_context.return_value = MockContext()

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x):
            out = all_reduce("sum", x, scale=1.0 / world_size)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)

    module = compile_model(Model(), [x], jit_script=False)

    text = raf._ffi.ir.AsText(module)
    assert text.count("_allreduce") == 1
    if world_size != 1:
        assert text.count("divide") == 1


@with_mock_distributed_context(world_size=4, rank=1)
def test_all_gather():
    """Test of tracing and lowering allgather op."""

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x):
            out = all_gather(x, dim=0)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)

    module = compile_model(Model(), [x], jit_script=False)

    text = raf._ffi.ir.AsText(module)
    ret_type = module["main"].ret_type
    expected_ret_shape = shape
    expected_ret_shape[0] *= 4

    assert text.count("_allgather") == 1
    assert list(ret_type.fields[1].shape) == expected_ret_shape


@with_enable_param_aliasing
@with_mock_distributed_context(world_size=4, rank=1)
def test_all_gather_out():
    """Test of tracing and lowering allgather op."""

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x, out):
            all_gather(x, dim=0, output=out)
            return x

    shape = [1, 1, 28, 28]
    expected_ret_shape = shape.copy()
    expected_ret_shape[0] *= 4

    x = torch.randn(*shape)
    out = torch.zeros(*expected_ret_shape).to("lazy")

    module = compile_model(Model(), [x, out], jit_script=False)
    text = raf._ffi.ir.AsText(module)
    assert text.count("_allgather") == 1


if __name__ == "__main__":
    pytest.main([__file__])
