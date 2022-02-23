"""Test torchvision models."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import mnm
import pytest
import torch.optim as optim
import torchvision
from torch_mnm.optimizer import LANS, SGD
from torch_mnm.testing import TorchLeNet, fake_image_dataset, train, verify
from torch_mnm.testing import with_seed, with_temp_cache, dryrun_dumped_ir_file


@pytest.mark.parametrize("optimizer", [optim.SGD, SGD, LANS])
@with_seed(0)
def test_lenet_cifar10(optimizer):
    """Test LeNet with CIFAR-10 and PyTorch optimizers."""
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    lazy_results = train("lazy", model, dataset, optimizer=optimizer, batch_size=batch_size)
    cpu_results = train("cpu", model, dataset, optimizer=optimizer, batch_size=batch_size)
    verify(lazy_results, cpu_results, tol=1e-3)


@pytest.mark.skip(reason="Blocked by https://github.com/pytorch/pytorch/pull/73197")
@pytest.mark.parametrize("amp", [False, True])
@with_seed(0)
def test_resnet18_imagenet(amp):
    """Test ResNet-18 with ImageNet and PyTorch SGD."""
    if amp and os.environ.get("RAZOR_DEVICE", None) != "GPU":
        pytest.skip("AMP requires GPU")

    batch_size = 1
    num_classes = 100
    dataset = fake_image_dataset(batch_size, 3, 224, num_classes)
    model = torchvision.models.resnet18(num_classes=num_classes)
    lazy_results = train(
        "lazy", model, dataset, batch_size=batch_size, amp=amp, num_classes=num_classes
    )
    cpu_results = train("cpu", model, dataset, batch_size=batch_size, num_classes=num_classes)
    verify(lazy_results, cpu_results, tol=1e-3)


@patch("mnm.distributed.get_context")
@with_temp_cache
@dryrun_dumped_ir_file
def test_compile_lenet_dp(mock_get_context):
    # Mock the dist context.
    class MockContext:
        def __init__(self):
            # Note that we do not use AutoDataParallel but manually all-reduce the gradients.
            self.enable_data_parallel = False
            self.zero_opt_level = 0
            self.size = 4
            self.rank = 0

    mock_get_context.return_value = MockContext()

    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()

    train(
        "lazy",
        model,
        dataset,
        optimizer=SGD,
        batch_size=batch_size,
        num_epochs=1,
        reduce_gradients=True,
    )
    meta_ir_file = os.environ["TORCH_MNM_SAVE_IR_FILE"]
    with open(meta_ir_file) as module_file:
        module_json = module_file.read()
        module = mnm.ir.serialization.LoadJSON(module_json)

    text = mnm.ir.AsText(module)
    assert text.count("_allreduce") == 8


if __name__ == "__main__":
    pytest.main([__file__])
