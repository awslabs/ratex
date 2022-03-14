# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import raf
from razor.testing import compile_model


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


if __name__ == "__main__":
    pytest.main([__file__])
