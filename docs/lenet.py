# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import ratex
import ratex.lazy_tensor_core.debug.metrics as metrics
import ratex.lazy_tensor_core.core.lazy_model as lm


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.relu(out)  # pylint: disable=no-member
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)  # pylint: disable=no-member
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out


def train(device, model, image_datasets):
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1, shuffle=False, num_workers=1
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    model = model.to(device, dtype=torch.float32)
    model.train()
    criterion = lambda pred, true: nn.functional.nll_loss(nn.LogSoftmax(dim=-1)(pred), true)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 10
    best_acc = 0.0
    unscripted = model
    if device == "lazy":
        model = ratex.jit.script(model)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels_one_hot = torch.tensor(np.eye(10, dtype=np.float32)[labels])
            labels_one_hot = labels_one_hot.to(device)  # One-hot
            optimizer.zero_grad()
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # loss = criterion(outputs, labels)
            # adapting loss cacluation from
            # https://www.programmersought.com/article/86167037001/
            # this doesn't match nn.NLLLoss() exactly, but close...
            loss = -torch.sum(outputs * labels_one_hot) / inputs.size(0)
            loss.backward()
            optimizer.step()
            lm.mark_step()
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = 0
        print("{} Loss: {:.4f}".format("train", epoch_loss))


def infer(device, model, image_datasets):
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1, shuffle=False, num_workers=1
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    model = model.to(device)
    model.eval()
    criterion = lambda pred, true: nn.functional.nll_loss(nn.LogSoftmax(dim=-1)(pred), true)
    best_acc = 0.0

    if device == "lazy":
        model = ratex.jit.script(model)
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        # running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes["val"]
    # epoch_acc = running_corrects.double() / dataset_sizes["train"]
    epoch_acc = 0
    print("{} Loss: {:.4f} Acc: {:.4f}".format("val", epoch_loss, epoch_acc))


def main():
    model_mnm = TorchLeNet()
    model_cpu = copy.deepcopy(model_mnm)
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
    print("raf starts...")
    train("lazy", model_mnm, image_datasets)
    print("cpu starts...")
    train("cpu", model_cpu, image_datasets)

    # print("raf starts...")
    # infer("raf", model_mnm, image_datasets)
    # print("cpu starts...")
    # infer("cpu", model_cpu, image_datasets)

    # statistics
    print(metrics.metrics_report())


if __name__ == "__main__":
    main()
