# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test ZeRO-1 implementation using FSDP interface."""
import pytest

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from raf.testing import get_dist_comm_info, skip_dist_test
from raf import distributed as dist

import ratex
import ratex.lazy_tensor_core.core.lazy_model as lm
from ratex.core.distributed.ratex_fully_sharded_data_parallel import RatexFullyShardedDataParallel
from ratex.optimizer import SGD, Adam
from ratex.testing import check
import numpy as np


class SingleLayerLogistics(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(SingleLayerLogistics, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.linear = nn.Linear(784, num_classes)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.linear(out)
        out = torch.relu(out)
        out = self.log_softmax(out)
        return out


def train(
    device,
    model,
    optimizer,
    optimizer_config,
    image_datasets,
    fsdp=False,
    num_epochs=10,
    dtype=torch.float32,
):
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1, shuffle=False, num_workers=1
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

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
            labels_one_hot = torch.tensor(np.eye(10, dtype=np.float32)[labels])
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
@pytest.mark.parametrize(
    "optimizer", [(SGD, {"lr": 0.001, "momentum": 0.1}, 1), (Adam, {"lr": 0.001}, 2)]
)
def test_ratex_fully_sharded_data_parallelism_zero1(optimizer, tolerance=1e-10, seed=0):
    torch.manual_seed(seed)
    model = SingleLayerLogistics()
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.CenterCrop(28),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(28),
                transforms.ToTensor(),
            ]
        ),
    }
    image_datasets = {
        x: datasets.FakeData(
            size=1, image_size=(1, 28, 28), num_classes=10, transform=data_transforms[x]
        )
        for x in ["train", "val"]
    }

    dcfg = dist.get_config()
    dcfg.zero_opt_level = 1
    total_rank, rank, local_rank = get_dist_comm_info()
    device = lm.lazy_device(rank)
    optimizer_zero1_loss = train(device, model, optimizer[0], optimizer[1], image_datasets)

    torch.manual_seed(seed)
    dcfg.zero_opt_level = 0
    model = SingleLayerLogistics()
    no_zero1_loss = train(device, model, optimizer[0], optimizer[1], image_datasets)

    check(no_zero1_loss, optimizer_zero1_loss, atol=tolerance)

    torch.manual_seed(seed)
    model = SingleLayerLogistics()
    fsdp_zero1_loss = train(
        device, model, optimizer[0], optimizer[1], image_datasets, fsdp=True
    )

    check(fsdp_zero1_loss, optimizer_zero1_loss, atol=tolerance)


if __name__ == "__main__":
    pytest.main([__file__])
