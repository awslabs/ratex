# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from ratex.testing import verify_step


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm(dtype, p, dim, keepdim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.norm(x_input, p, dim, keepdim, dtype=dtype)

    x = torch.rand(2, 3).to(dtype)

    verify_step(Model(), [x], jit_script=False, tol=(1e-3 if dtype == torch.float16 else 1e-5))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_any(dim, keepdim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.any(x_input, dim, keepdim)

    x = torch.rand(2, 3).bool()

    verify_step(Model(), [x], jit_script=False)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_max(dim, keepdim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.max(x_input, dim, keepdim)

    x = torch.rand(2, 3)

    verify_step(Model(), [x], jit_script=False)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_argmax(dim, keepdim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.argmax(x_input, dim, keepdim)

    x = torch.rand(2, 3)

    verify_step(Model(), [x], jit_script=False)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(dtype, dim, keepdim):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.mean(x_input, dim, keepdim, dtype=dtype)

    x = torch.rand(4, 4)

    verify_step(Model(), [x], jit_script=False)


if __name__ == "__main__":
    pytest.main([__file__])
