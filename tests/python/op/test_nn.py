import copy
import time
import os

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torch_mnm
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as lm

def run(device, model_origin, args):
    model = copy.deepcopy(model_origin)
    model = model.to(device, dtype=torch.float32)
    if device == "xla":
        model = torch_mnm.jit.script(model)
    args = [arg.to(device) for arg in args]
    return model(*args).to("cpu")


def verify(model, args):
    torch.testing.assert_close(
        run("cpu", model, args),
        run("xla", model, args)
    )


def test_conv():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.conv = nn.Conv2d(in_channels=1,
                                  out_channels=6,
                                  kernel_size=5,
                                  padding=2,
                                  bias=False)

        def forward(self, x):
            out = self.conv(x)
            return out

    shape = [1, 1, 28, 28]
    x = torch.randn(*shape)
    verify(Test(), [x])


def test_linear():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.linear = nn.Linear(120, 84)

        def forward(self, x):
            out = self.linear(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify(Test(), [x])


def test_sum():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()

        def forward(self, x):
            out = torch.sum(x)
            return out

    shape = [32, 120]
    x = torch.randn(*shape)
    verify(Test(), [x])


if __name__ == "__main__":
    pytest.main([__file__])
