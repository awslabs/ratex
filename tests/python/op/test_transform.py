# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pytest

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import ratex
from ratex.lazy_tensor_core.core import lazy_model as lm
from ratex.testing import verify_step


@pytest.mark.xfail(reason="Need to fix the accuracy issue")
def test_embedding():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32003, 16)

        def forward(self, x):
            out = self.embedding(x)
            out = torch.sum(out)
            return out

    def step(device, model_origin, args):
        model = copy.deepcopy(model_origin)
        model.train()
        tmp = model
        if device == "lazy":
            model = ratex.jit.script(model)
        model = model.to(device, dtype=torch.float32)
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        args = [arg.to(device) for arg in args]
        loss = model(*args)
        loss.backward()
        optimizer.step()
        lm.mark_step()
        return loss.to("cpu"), torch.Tensor(tmp.embedding.weight.to("cpu"))

    x = torch.LongTensor([[0, 1, 2, 3], [4, 5, 6, 0]])
    model = Model()
    loss_cpu, grad_cpu = step("cpu", model, [x])
    loss_ltc, grad_ltc = step("lazy", model, [x])
    torch.testing.assert_close(loss_cpu, loss_ltc)
    torch.testing.assert_close(grad_cpu, grad_ltc)


def test_select():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = x[2:4]
            out = torch.sum(out)
            return out

    def step(device, model_origin, args):
        model = copy.deepcopy(model_origin)
        model = model.to(device, dtype=torch.float32)
        model.train()
        args = [arg.to(device) for arg in args]
        loss = model(*args)
        loss.backward()
        lm.mark_step()
        return loss.to("cpu")

    shape = (5, 4, 4)
    model = Model()
    n_x = np.random.randn(*shape)
    t_x_cpu = torch.tensor(n_x, device="cpu", dtype=torch.float32, requires_grad=True)
    t_x_ratex = torch.tensor(n_x, device="lazy", dtype=torch.float32, requires_grad=True)

    loss_cpu = step("cpu", model, [t_x_cpu])
    loss_ltc = step("lazy", model, [t_x_ratex])

    torch.testing.assert_close(loss_cpu, loss_ltc)
    torch.testing.assert_close(t_x_cpu.grad, t_x_ratex.grad.to("cpu"))


def test_scatter():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input, x_index, x_src):
            return torch.scatter(x_input, dim=0, index=x_index, src=x_src)

    input_shape = (5, 4)
    src_shape = (3, 4)
    x_input = torch.randn(*input_shape)
    x_src = torch.randn(*src_shape)
    x_index = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
    x_index = torch.from_numpy(x_index)
    verify_step(Model(), [x_input, x_index, x_src], jit_script=False)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_stack(dim, dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *tensors):
            return torch.stack(tuple(tensors), dim)

    shape = [3, 4]
    x = torch.randn(*shape).to(dtype)
    y = torch.randn(*shape).to(dtype)
    z = torch.randn(*shape).to(dtype)

    verify_step(Model(), [x, y, z], jit_script=False)


@pytest.mark.parametrize("chunks", [2, 3])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunk(chunks, dim, dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.chunk(x_input, chunks, dim)

    x = torch.randn(6, 6).to(dtype)
    verify_step(Model(), [x], jit_script=False)


@pytest.mark.parametrize("split_sizes", [[1, 2], [2, 1]])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_split(split_sizes, dim, dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_input):
            return torch.split(x_input, split_sizes, dim)

    x = torch.randn(3, 3).to(dtype)
    verify_step(Model(), [x], jit_script=False)


if __name__ == "__main__":
    pytest.main([__file__])
