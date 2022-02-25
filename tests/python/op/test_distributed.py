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
from razor.testing import compile_model


@patch("raf.distributed.get_context")
@pytest.mark.parametrize("world_size", [1, 4])
def test_allreduce(mock_get_context, world_size):
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


def test_allgather():
    """Test of tracing and lowering allgather op."""

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x):
            out = all_gather(x, dim=0)
            return out

    dctx = dist.get_context()
    dctx.rank = 1
    dctx.size = 4
    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)

    module = compile_model(Model(), [x], jit_script=False)

    text = raf._ffi.ir.AsText(module)
    ret_type = module["main"].ret_type
    expected_ret_shape = shape
    expected_ret_shape[0] *= dctx.size

    assert text.count("_allgather") == 1
    assert list(ret_type.shape) == expected_ret_shape


if __name__ == "__main__":
    pytest.main([__file__])
