# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

import ratex
from ratex.testing import verify_step


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

        def forward(self, x):
            out = torch.nn.GELU("none")(x)
            return out

    shape = [5, 5]
    x = torch.randn(*shape)
    verify_step(Model(), [x])


def test_embedding():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 3)

        def forward(self, x_input):
            return self.embedding(x_input)

    x = torch.randint(10, (3, 3))
    verify_step(Model(), [x], jit_script=False)


if __name__ == "__main__":
    pytest.main([__file__])
