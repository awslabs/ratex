# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from razor.testing import verify_step


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


if __name__ == "__main__":
    pytest.main([__file__])
