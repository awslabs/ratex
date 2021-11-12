"""Test torchvision models."""
import os

import pytest
import torchvision

import torch.optim as optim
from torch_mnm.optimizer import LANS

from torch_mnm.testing import TorchLeNet, fake_image_dataset, verify, train, with_seed


@pytest.mark.parametrize("optimizer", [optim.SGD, LANS])
@with_seed(0)
def test_lenet(optimizer):
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    xla_results = train("xla", model, dataset, optimizer=optimizer, batch_size=batch_size)
    cpu_results = train("cpu", model, dataset, optimizer=optimizer, batch_size=batch_size)
    verify(xla_results, cpu_results, tol=1e-3)


@pytest.mark.parametrize("amp", [False, True])
@with_seed(0)
def test_resnet18(amp):
    if amp and os.environ.get("RAZOR_DEVICE", None) != "GPU":
        pytest.skip("AMP requires GPU")

    batch_size = 1
    dataset = fake_image_dataset(batch_size, 3, 224, 100)
    model = torchvision.models.resnet18()
    xla_results = train("xla", model, dataset, batch_size=batch_size, amp=amp)
    cpu_results = train("cpu", model, dataset, batch_size=batch_size)
    verify(xla_results, cpu_results, tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
