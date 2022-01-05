"""Test torchvision models."""
import os

import pytest
import torchvision

import torch.optim as optim
from torch_mnm.optimizer import LANS, SGD
from torch_mnm.testing import TorchLeNet, fake_image_dataset, verify, train, with_seed


@pytest.mark.parametrize("optimizer", [optim.SGD, SGD, LANS])
@with_seed(0)
def test_lenet_cifar10(optimizer):
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    xla_results = train("xla", model, dataset, optimizer=optimizer, batch_size=batch_size)
    cpu_results = train("cpu", model, dataset, optimizer=optimizer, batch_size=batch_size)
    verify(xla_results, cpu_results, tol=1e-3)


@pytest.mark.parametrize("momentum", [0.0, 0.1])
@with_seed(0)
def test_lenet_cifar10_sgd_razor_vs_torch(momentum):
    """
    This test uses Pytorch SGD as golden result, compared with the Razor SGD implementation.
    """
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    optimizer_params = {"lr": 0.01, "momentum": momentum}
    xla_results = train(
        "xla",
        model,
        dataset,
        optimizer=SGD,
        optimizer_params=optimizer_params,
        batch_size=batch_size,
        trim=True,
    )
    cpu_results = train(
        "cpu",
        model,
        dataset,
        optimizer=optim.SGD,
        optimizer_params=optimizer_params,
        batch_size=batch_size,
    )
    verify(xla_results, cpu_results, tol=1e-2)


@pytest.mark.parametrize("amp", [False, True])
@with_seed(0)
def test_resnet18_imagenet(amp):
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
