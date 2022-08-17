# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test ZeRO-1 implementation using FSDP interface."""
import pytest

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from raf.testing import with_seed, get_dist_comm_info, skip_dist_test
from raf import distributed as dist

import ratex
import ratex.lazy_tensor_core.core.lazy_model as lm
from ratex.core.distributed.ratex_fully_sharded_data_parallel import RatexFullyShardedDataParallel
from ratex.optimizer import SGD, Adam
from ratex.testing import check
import numpy as np


class SingleLayerLogistics(nn.Module):
    def __init__(self, input_shape=28, num_classes=11):
        super(SingleLayerLogistics, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.linear = nn.Linear(input_shape**2, num_classes)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.linear(out)
        out = torch.relu(out)
        out = self.log_softmax(out)
        return out


def train(
    device,
    input_shape,
    num_classes,
    optimizer,
    optimizer_config,
    image_datasets,
    fsdp=False,
    num_epochs=10,
    dtype=torch.float32,
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1, shuffle=False, num_workers=1
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    model = SingleLayerLogistics(input_shape=input_shape, num_classes=num_classes)
    model = ratex.jit.script(model)
    model = model.to(device, dtype=dtype)
    if fsdp:
        model = RatexFullyShardedDataParallel(model, optimizer, optimizer_config)
        optimizer = model
        model.train()
    else:
        model.train()
        optimizer = optimizer(model.parameters(), **optimizer_config)

    for epoch in range(num_epochs):
        running_losses = []
        # Iterate over data.
        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels_one_hot = torch.tensor(np.eye(num_classes, dtype=np.float32)[labels])
            labels_one_hot = labels_one_hot.to(device)  # One-hot
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = -torch.sum(outputs * labels_one_hot) / inputs.size(0)
            loss.backward()
            optimizer.step()
            lm.mark_step()
            running_losses.append((loss, inputs.size(0)))

        epoch_loss = (
            sum([loss.item() * size for loss, size in running_losses]) / dataset_sizes["train"]
        )
    return epoch_loss


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason="ZeRO-1 requires multiple GPU's")
@pytest.mark.parametrize("input_shape", [28])
@pytest.mark.parametrize("num_classes", [12, 13])
@pytest.mark.parametrize(
    "optimizer", [(SGD, {"lr": 0.001, "momentum": 0.1}), (Adam, {"lr": 0.001})]
)
def test_ratex_fully_sharded_data_parallelism_zero1(
    input_shape, num_classes, optimizer, tolerance=1e-10, seed=0
):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.CenterCrop(input_shape),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(input_shape),
                transforms.ToTensor(),
            ]
        ),
    }
    image_datasets = {
        x: datasets.FakeData(
            size=1,
            image_size=(1, input_shape, input_shape),
            num_classes=num_classes,
            transform=data_transforms[x],
        )
        for x in ["train", "val"]
    }

    dcfg = dist.get_config()
    total_rank, rank, local_rank = get_dist_comm_info()
    device = lm.lazy_device(rank)

    dcfg.zero_opt_level = 0
    no_zero1_loss = train(
        device,
        input_shape,
        num_classes,
        optimizer[0],
        optimizer[1],
        image_datasets,
        seed=seed,
    )

    # Error in optimizer ZeRO-1 when tensor padding is required
    if num_classes % total_rank == 0:
        dcfg.zero_opt_level = 1
        optimizer_zero1_loss = train(
            device,
            input_shape,
            num_classes,
            optimizer[0],
            optimizer[1],
            image_datasets,
            seed=seed,
        )

        check(no_zero1_loss, optimizer_zero1_loss, atol=tolerance)

    dcfg.zero_opt_level = 0
    fsdp_zero1_loss = train(
        device,
        input_shape,
        num_classes,
        optimizer[0],
        optimizer[1],
        image_datasets,
        fsdp=True,
        seed=seed,
    )

    check(no_zero1_loss, fsdp_zero1_loss, atol=tolerance)


if __name__ == "__main__":
    pytest.main([__file__])
