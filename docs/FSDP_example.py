# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import ratex
import ratex.lazy_tensor_core.core.lazy_model as lm
from ratex.lazy_tensor_core.core.lazy_model import lazy_device
import ratex.optimizer as optim
from ratex.core.distributed.ratex_fully_sharded_data_parallel import RatexFullyShardedDataParallel

from raf import distributed as dist
from raf.testing import get_dist_comm_info

import numpy as np


# Optional to record IR graph
# import _RATEXC
# _RATEXC._set_ratex_vlog_level(-5)


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


def train(device, model, image_datasets, num_epochs=10):
    # Data Setup
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1, shuffle=False, num_workers=1
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    # Optimizer setup
    optimizer_init = optim.SGD
    optimizer_config = {"lr": 0.001, "momentum": 0.1}

    # Model Setup
    model = ratex.jit.script(model)
    model = model.to(device, dtype=torch.float32)

    # Ratex FSDP Wrapper
    model = RatexFullyShardedDataParallel(model, optimizer_init, optimizer_config)

    model.train()
    unscripted = model

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        running_losses = []
        # Iterate over data.
        for inputs, labels in dataloaders["train"]:

            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels_one_hot = torch.tensor(np.eye(10, dtype=np.float32)[labels])
            labels_one_hot = labels_one_hot.to(device)  # One-hot

            # Zero out the gradients through the model
            unscripted.zero_grad()

            outputs = model(inputs)
            loss = -torch.sum(outputs * labels_one_hot) / inputs.size(0)
            loss.backward()

            # Step the optimizer through the model
            unscripted.step()

            lm.mark_step()
            running_losses.append((loss, inputs.size(0)))

        epoch_loss = (
            sum([loss.item() * size for loss, size in running_losses]) / dataset_sizes["train"]
        )
        print("{} Loss: {:.4f}".format("train", epoch_loss))


def main():
    torch.manual_seed(0)
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

    # Optimizer ZeRO is not needed because of the FSDP interface is used instead
    dcfg.zero_opt_level = 0
    total_rank, rank, local_rank = get_dist_comm_info()

    print("raf starts...")
    train(lazy_device(rank), model, image_datasets)


if __name__ == "__main__":
    main()
