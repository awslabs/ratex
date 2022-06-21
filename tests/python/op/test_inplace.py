# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import raf
import numpy as np
from ratex.testing import compile_model, run_step, with_enable_param_aliasing


def test_mul_():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x):
            x.mul_(2.0)
            return x

    x = torch.randn(2, 2)
    module = compile_model(Model(), [x], jit_script=False)
    text = raf._ffi.ir.AsText(module)
    assert text.count("cast") == 0
    assert text.count("multiply") == 1


def test_div():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x):
            return x / 2.0

    x = torch.randn(2, 2)
    module = compile_model(Model(), [x], jit_script=False)
    text = raf._ffi.ir.AsText(module)
    assert text.count("cast") == 0
    assert text.count("divide") == 1


@with_enable_param_aliasing
def test_mul_out():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x, out):
            torch.multiply(x, 2.0, out=out)
            return x

    x_np = np.random.rand(3, 2)
    x_t = torch.from_numpy(x_np)
    out_t = torch.from_numpy(x_np).to("lazy")
    run_step("lazy", Model(), [x_t, out_t], jit_script=False)
    out_np = out_t.to("cpu").numpy()
    torch.testing.assert_close(out_t.to("cpu").numpy(), x_np * 2.0)


if __name__ == "__main__":
    pytest.main([__file__])
