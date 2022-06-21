# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=abstract-method, arguments-differ, attribute-defined-outside-init, no-member
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import raf
from raf.model import Conv2d, Linear, BatchNorm
from raf.testing import run_vm_model, one_hot_torch, randn_torch, t2m_param, with_seed

from ratex.optimizer import LANS
from ratex.testing import check


class TorchSimpleTest(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(*shape))
        self.x.requires_grad = True

    def forward(self):
        y = F.relu(self.x)
        return y


class RAFSimpleTest(raf.Model):
    def build(self, shape):
        self.x = raf.array(np.random.randn(*shape).astype("float32"))

    @raf.model.trace
    def forward(self):
        y = raf.relu(self.x)
        return y


class TorchTest(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchTest, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.linear1 = nn.Linear((input_shape // 2) ** 2 * 6, num_classes)

    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = F.log_softmax(y_pred, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss

    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = torch.relu(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        return out


class RAFTest(raf.Model):
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @raf.model.trace
    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = raf.relu(out)
        out = raf.avg_pool2d(out, (2, 2), (2, 2))
        out = raf.batch_flatten(out)
        out = self.linear1(out)
        return out


@pytest.mark.skipif(
    not raf.build.with_cuda(), reason="Require CUDA to run LANS kernel for comparison"
)
@with_seed(0)
def test_traced_lans_simple():
    t_device = "cpu"
    m_device = "cuda"  # lans kernel only available on cuda
    shape = (2, 2)
    iter_size = 4
    t_model = TorchSimpleTest(shape)
    t_model.train()
    t_model.to(t_device)
    t_optimizer = LANS(t_model.parameters())
    m_model = RAFSimpleTest(shape)
    m_model.x = t2m_param(t_model.x, device=m_device)
    m_model.train_mode()
    m_optimizer = raf.optim.lans.with_lans()(m_model)
    for _ in range(iter_size):
        m_dy, t_dy = randn_torch(shape, device=t_device, requires_grad=False)
        m_dy = m_dy.to(device=m_device)
        run_vm_model(m_optimizer, m_device, [m_dy])
        t_optimizer.zero_grad()
        t_loss = t_model()
        t_loss.backward(t_dy)
        t_optimizer.step()
        check(m_model.x, t_model.x, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not raf.build.with_cuda(), reason="Require CUDA to run LANS kernel for comparison"
)
@pytest.mark.parametrize(
    "config",
    [
        (4, 28, 10),
    ],
)
@with_seed(0)
def test_traced_lans(config):
    # pylint: disable=too-many-locals
    tol = 5e-4
    t_device = "cpu"
    m_device = "cuda"  # lans kernel only available on cuda
    iter_size = config[0]
    t_model = TorchTest(config[1], config[2])
    t_model.to(device=t_device)
    m_model = RAFTest(config[1], config[2])
    m_model.to(device=m_device)
    m_model.conv1.w = t2m_param(t_model.conv1.weight, device=m_device)
    m_model.linear1.w = t2m_param(t_model.linear1.weight, device=m_device)
    m_model.linear1.b = t2m_param(t_model.linear1.bias, device=m_device)
    m_model.bn1.w = t2m_param(t_model.bn1.weight, device=m_device)
    m_model.bn1.b = t2m_param(t_model.bn1.bias, device=m_device)
    m_model.bn1.running_mean = t2m_param(t_model.bn1.running_mean, device=m_device)
    m_model.bn1.running_var = t2m_param(t_model.bn1.running_var, device=m_device)

    m_model.train_mode()
    t_model.train()
    m_optimizer = raf.optim.lans.with_lans()(m_model)
    t_optimizer = LANS(t_model.parameters())
    for _ in range(iter_size):
        m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, device=t_device, requires_grad=False)
        m_dy = m_dy.to(device=m_device)
        m_x, t_x = randn_torch([1, 3, config[1], config[1]], requires_grad=True, device=t_device)
        m_x = m_x.to(device=m_device)
        m_y, t_y = one_hot_torch(size=1, num_classes=config[2], device=t_device)
        m_y = m_y.to(device=m_device)
        run_vm_model(m_optimizer, m_device, [m_dy, m_x, m_y])
        t_optimizer.zero_grad()
        t_loss = t_model(t_x, t_y)
        t_loss.backward(t_dy)
        t_optimizer.step()
        check(m_model.conv1.w, t_model.conv1.weight, rtol=tol, atol=tol)
        check(m_model.linear1.w, t_model.linear1.weight, rtol=tol, atol=tol)
        check(m_model.linear1.b, t_model.linear1.bias, rtol=tol, atol=tol)
        check(m_model.bn1.w, t_model.bn1.weight, rtol=tol, atol=tol)
        check(m_model.bn1.b, t_model.bn1.bias, rtol=tol, atol=tol)


if __name__ == "__main__":
    pytest.main([__file__])
