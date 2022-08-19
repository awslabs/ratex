# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test torchvision models."""
import os
from unittest.mock import patch

import pytest
import raf
import torch.optim as optim
import torchvision
from raf.testing import check
from ratex.optimizer import LANS, SGD, Adam
from ratex.testing import (
    TorchLeNet,
    dryrun_dumped_ir_file,
    fake_image_dataset,
    get_most_recent_alias,
    train,
    verify,
    with_enable_param_aliasing,
    with_mock_distributed_info,
    with_seed,
    with_temp_cache,
)
from ratex.utils.utils import to_torch_name

LENET_PARAM_NUM = 8


@pytest.mark.parametrize("optimizer", [optim.SGD, SGD, LANS])
@with_seed(0)
def test_lenet_cifar10(optimizer):
    """Test LeNet with CIFAR-10 and PyTorch optimizers."""
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    lazy_results, lazy_model = train(
        "lazy", model, dataset, optimizer=optimizer, batch_size=batch_size, return_model=True
    )
    cpu_results = train("cpu", model, dataset, optimizer=optimizer, batch_size=batch_size)

    # Verify the loss
    verify(lazy_results, cpu_results, tol=1e-3)

    # Verify the parameter mapping for inference.
    lazy_params = {k: v.cpu() for k, v in lazy_model.named_parameters()}
    model = lazy_model.native_cpu()
    model_dict = model.state_dict()
    for raf_name, val in lazy_params.items():
        torch_name = to_torch_name(raf_name)
        if torch_name in model_dict:
            check(model_dict[torch_name], val)


@pytest.mark.parametrize("amp", [False, True])
@with_seed(0)
def test_resnet18_imagenet(amp):
    """Test ResNet-18 with ImageNet and PyTorch SGD."""
    if amp and os.environ.get("RATEX_DEVICE", None) != "GPU":
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


@patch("raf.distributed.get_communicator")
@patch("raf.distributed.get_config")
@with_temp_cache
@dryrun_dumped_ir_file
def test_compile_lenet_dp(mock_get_config, mock_get_comm):
    # Mock the dist config and communicator.
    class MockConfig:
        def __init__(self):
            self.enable_data_parallel = False
            self.zero_opt_level = 0

    mock_get_config.return_value = MockConfig()

    class MockComm:
        def __init__(self):
            self.size = 4
            self.rank = 0

    mock_get_comm.return_value = MockComm()

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
    meta_ir_file = os.environ["RATEX_SAVE_IR_FILE"]
    with open(meta_ir_file) as module_file:
        module_json = module_file.read()
        module = raf.ir.serialization.LoadJSON(module_json)

    text = raf.ir.AsText(module)
    assert text.count("_allreduce") == LENET_PARAM_NUM


@with_temp_cache
@dryrun_dumped_ir_file
@with_enable_param_aliasing
@with_mock_distributed_info(world_size=2, rank=1, zero_opt_level=1)
@pytest.mark.parametrize(
    "optimizer", [(SGD, {"lr": 0.001, "momentum": 0.1}, 1), (Adam, {"lr": 0.001}, 2)]
)
@pytest.mark.parametrize("grad_inplace", [True, False])
def test_compile_lenet_zero1(optimizer, grad_inplace):
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()

    train(
        "lazy",
        model,
        dataset,
        optimizer=optimizer[0],
        optimizer_params=optimizer[1],
        batch_size=batch_size,
        num_epochs=1,
        trim=True,
        set_to_none=not grad_inplace,
    )

    # Last RAF IR graph is the optimizer graph
    meta_ir_file = os.environ["RATEX_SAVE_IR_FILE"]
    with open(meta_ir_file) as module_file:
        module_json = module_file.read()
        module = raf.ir.serialization.LoadJSON(module_json)

    text = raf.ir.AsText(module)
    alias = get_most_recent_alias()
    assert text.count("_allgather") == LENET_PARAM_NUM

    # If gradient is set to zero, expected alias num = #weights + #model_states
    # If not, expected alias num = #grads + #weights + #model_states
    expected_alias_num = LENET_PARAM_NUM * (grad_inplace + 1 + optimizer[2])
    assert len(alias) >= expected_alias_num


if __name__ == "__main__":
    pytest.main([__file__])
