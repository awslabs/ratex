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
    dataloaders = torch.utils.data.DataLoader(
        image_datasets, batch_size=1, shuffle=False, num_workers=1
    )
    dataset_size = len(image_datasets)
    model.train()
    criterion = lambda pred, true: nn.functional.nll_loss(nn.LogSoftmax(dim=-1)(pred), true)
    num_epochs = 10
    best_acc = 0.0

    if device == "lazy":
        model = ratex.jit.script(model)
    model = model.to(device, dtype=torch.float32)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
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

        epoch_loss = running_loss / dataset_size
        epoch_acc = 0
        print("{} Loss: {:.4f}".format("train", epoch_loss))
    return model


def infer(device, model, image_datasets):
    dataloader = torch.utils.data.DataLoader(
        image_datasets, batch_size=1, shuffle=False, num_workers=1
    )
    dataset_size = len(image_datasets)
    model.eval()
    model = model.to(device)

    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs)
        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_size
    print("{} Acc: {:.4f}".format("test", acc))


def main():
    model_raf = TorchLeNet()
    model_cpu = copy.deepcopy(model_raf)
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.CenterCrop(28),
                transforms.ToTensor(),
            ]
        ),
        "test": transforms.Compose(
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
        for x in ["train", "test"]
    }
    print("raf starts...")
    model_raf = train("lazy", model_raf, image_datasets["train"])
    print("cpu starts...")
    train("cpu", model_cpu, image_datasets["train"])

    print("raf starts...")
    infer("cpu", model_raf.native_cpu(), image_datasets["test"])

    # statistics
    print(metrics.metrics_report())


if __name__ == "__main__":
    main()
