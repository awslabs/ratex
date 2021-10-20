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


def step(device, model_origin, args):
    model = copy.deepcopy(model_origin)
    model = model.to(device, dtype=torch.float32)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    tmp = model
    if device == "xla":
        model = torch_mnm.jit.script(model)
    args = [arg.to(device) for arg in args]
    loss = model(*args)
    loss.backward()
    optimizer.step()
    lm.mark_step()
    return loss.to("cpu"), torch.Tensor(tmp.embedding.weight.to("cpu"))


def test_embedding():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.embedding = nn.Embedding(32003, 16)

        def forward(self, x):
            out = self.embedding(x)
            out = torch.sum(out)
            return out

    batch_size = 2
    seq_length = 4
    x = torch.LongTensor([[0, 1, 2, 3], [4, 5, 6, 0]])
    model = Test()
    loss_cpu, grad_cpu = step("cpu", model, [x])
    loss_ltc, grad_ltc = step("xla", model, [x])
    torch.testing.assert_close(loss_cpu, loss_ltc)
    torch.testing.assert_close(grad_cpu, grad_ltc)


if __name__ == "__main__":
    pytest.main([__file__])
