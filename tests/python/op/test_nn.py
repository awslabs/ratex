# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from ratex.lazy_tensor_core.core import lazy_model as lm
import numpy as np
import ratex
from ratex.testing import verify_step, check


def test_conv():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False
            )

        def forward(self, x):
            out = self.conv(x)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)
    verify_step(Model(), [x])


def test_linear():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(120, 84)

        def forward(self, x):
            out = self.linear(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify_step(Model(), [x])


def test_sum():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = torch.sum(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify_step(Model(), [x], jit_script=False, tol=5e-4)


def test_pad():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            pad = (1, 2, 3, 4, 5, 6)
            out = torch.nn.functional.pad(x, pad, "constant", 2)
            return out

    shape = [32, 120, 20]
    x = torch.randn(*shape)
    verify_step(Model(), [x], jit_script=False)


def test_gelu():
    """GeLU supports approximation since https://github.com/pytorch/pytorch/pull/72826"""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.gelu = torch.nn.GELU("none")

        def forward(self, x):
            return self.gelu(x)

    shape = [5, 5]
    t_x_cpu = torch.randn(shape, requires_grad=True)

    verify_step(Model(), [t_x_cpu], jit_script=False)
    verify_step(Model(), [t_x_cpu], jit_script=False, with_backward=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("norm_type", [1, 2])
def test_embedding(dtype, norm_type):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 3, norm_type=norm_type, dtype=dtype)

        def forward(self, x_input):
            return self.embedding(x_input)

    x = torch.randint(10, (3, 3))
    verify_step(Model(), [x], jit_script=False)


def test_softmax():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax()

        def forward(self, x_input):
            return self.softmax(x_input)

    shape = (3, 3)
    t_x_cpu = torch.randn(shape, requires_grad=True)

    verify_step(Model(), [t_x_cpu], jit_script=False)
    verify_step(Model(), [t_x_cpu], jit_script=False, with_backward=True)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("p", [0.2, 0.6])
def test_dropout(p, dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=p)

        def forward(self, x_input):
            return self.dropout(x_input)

    def check_dropout(x, y, dx=None, dy=None):
        x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
        mask = y != 0
        expected = mask * x / (1 - p)
        check(expected, y)
        frac = np.sum(y == 0) / y.size
        assert p - 0.1 < frac < p + 0.1
        if dx is not None and dy is not None:
            dx, dy = dx.numpy(), dy.numpy()
            expected = mask / (1 - p) * dy
            check(expected, dx)

    shape = (128, 128)
    t_x_ratex = torch.randn(*shape, dtype=dtype, requires_grad=True).to("lazy")
    model = Model()
    out = model(t_x_ratex)

    check_dropout(t_x_ratex, out)

    d_x = np.random.randn(*shape)
    dy = torch.tensor(d_x, device="lazy", dtype=dtype, requires_grad=True)

    out.backward(dy)

    check_dropout(t_x_ratex, out, t_x_ratex.grad, dy)


if __name__ == "__main__":
    pytest.main([__file__])
