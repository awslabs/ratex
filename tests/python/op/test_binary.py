# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from ratex.testing import verify_step, run_step


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("op", [torch.mul, torch.div, torch.sub, torch.add])
def test_basic(op, dtype):
    class BinaryModel(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x1, x2):
            return self.op(x1, x2)

    shape = [3, 4]
    a = torch.randn(*shape).to(dtype)
    b = torch.randn(*shape).to(dtype)
    verify_step(BinaryModel(op), [a, b], jit_script=False)


@pytest.mark.parametrize("op", [torch.logical_or])
def test_logical(op):
    class BinaryModel(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x1, x2):
            return self.op(x1, x2)

    shape = [3, 4]
    a = torch.randn(*shape).bool()
    b = torch.randn(*shape).bool()
    verify_step(BinaryModel(op), [a, b], jit_script=False)


def test_basic_bool():
    # When use `add`, `alpha` will become a bool scalar here
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.add(x1, x2)

    shape = [3, 4]
    a = torch.randn(*shape).bool()
    b = torch.randn(*shape).bool()
    # not use `verify_step` as boolean addition is undefined behavior
    run_step("lazy", Model(), [a, b], jit_script=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("op", [torch.add, torch.sub])
def test_op_with_alpha(op, dtype):
    class BinaryModelWithAlpha(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x1, x2):
            return self.op(x1, x2, alpha=2.0)

    shape = [3, 4]
    a = torch.randn(*shape).to(dtype)
    b = torch.randn(*shape).to(dtype)
    verify_step(BinaryModelWithAlpha(op), [a, b], jit_script=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("op", [torch.lt, torch.gt, torch.eq, torch.ne])
def test_comparison(op, dtype):
    class BinaryModel(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x1, x2):
            return self.op(x1, x2)

    shape = [3, 4]
    a = torch.randn(*shape).to(dtype)
    b = torch.randn(*shape).to(dtype)
    verify_step(BinaryModel(op), [a, b], jit_script=False)


if __name__ == "__main__":
    pytest.main([__file__])
