import os

import pytest
import torch
import torch.nn as nn

import torch_mnm
from torch_mnm.testing import compile_only
from torch_mnm.core.lazy_model import all_reduce, all_gather

import mnm
from mnm import distributed as dist


@pytest.mark.parametrize("world_size", [1, 4])
def test_allreduce(world_size):
    """
    Test of tracing and lowering allreduce op.
    """

    dctx = dist.get_context()
    dctx.rank = 0
    dctx.size = world_size

    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()

        def forward(self, x):
            out = all_reduce("sum", x, scale=1.0 / dctx.size)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)

    module = compile_only(Test(), [x], jit_script=False)

    text = mnm._ffi.ir.AsText(module)
    assert text.count("_allreduce") == 1
    if world_size != 1:
        assert text.count("divide") == 1


def test_allgather():
    """
    Test of tracing and lowering allgather op.
    """

    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()

        def forward(self, x):
            out = all_gather(x, dim=0)
            return out

    dctx = dist.get_context()
    dctx.rank = 1
    dctx.size = 4
    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)

    module = compile_only(Test(), [x], jit_script=False)

    text = mnm._ffi.ir.AsText(module)
    ret_type = module["main"].ret_type
    expected_ret_shape = shape
    expected_ret_shape[0] *= dctx.size

    assert text.count("_allgather") == 1
    assert list(ret_type.shape) == expected_ret_shape


if __name__ == "__main__":
    pytest.main([__file__])
