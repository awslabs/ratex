# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from ratex.testing import verify_step


@pytest.mark.parametrize(
    "shape",
    [
        [2],
        [2, 3],
        [2, 3, 5],
    ],
)
def test_reciprocal(shape):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.reciprocal(x_input)

    x = torch.rand(shape)

    verify_step(Model(), [x], jit_script=False)


def test_isnan():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.isnan(x_input)

    x = torch.tensor([torch.nan, torch.inf, 0])

    verify_step(Model(), [x], jit_script=False)


if __name__ == "__main__":
    pytest.main([__file__])
