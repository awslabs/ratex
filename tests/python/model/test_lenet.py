import copy
import time
import os

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

import torch_mnm
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as lm
from torch_mnm.optimizer import LANS

class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               padding=2,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16,
                                 120)
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


def train(device, model, dataset, optimizer=optim.SGD, num_epochs=10):
    results = []
    model = copy.deepcopy(model)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=1)
    dataset_size = len(dataset)
    model = model.to(device, dtype=torch.float32)
    model.train()
    # adapting loss cacluation from
    # https://www.programmersought.com/article/86167037001/
    # this doesn't match nn.NLLLoss() exactly, but close...
    criterion = lambda pred, true: -torch.sum(pred * true) / true.size(0)
    optimizer = optimizer(model.parameters(), lr=0.001)
    if device == "xla":
        model = torch_mnm.jit.script(model)
    for epoch in range(num_epochs):
        # print('-' * 10)
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels_one_hot = torch.tensor(np.eye(10, dtype=np.float32)[labels], dtype=torch.float32)
            labels_one_hot = labels_one_hot.to(device) # One-hot
            optimizer.zero_grad()
            outputs = model(inputs)
            # No valid dispatch for op mnm.op.log_softmax on CPU
            # outputs = torch.nn.functional.log_softmax(outputs)
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()
            lm.mark_step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_size
        results.append(epoch_loss)
    return results


def verify(model, dataset, **kwargs):
    xla_results = train("xla", model, dataset)
    cpu_results = train("cpu", model, dataset)
    print("xla_results = ", xla_results)
    print("cpu_results = ", cpu_results)
    results = [torch.testing.assert_close(xla, cpu) for xla, cpu in zip(xla_results, cpu_results)]
    return results


def test_lenet():
    verify(
        TorchLeNet(),
        datasets.FakeData(size=1, image_size=(1, 28, 28), num_classes=10,
                          transform=transforms.ToTensor())
    )


def test_lenet_lans():
    verify(
        TorchLeNet(),
        datasets.FakeData(size=1, image_size=(1, 28, 28), num_classes=10,
                          transform=transforms.ToTensor()),
        optimizer=LANS,
    )


if __name__ == "__main__":
    pytest.main([__file__])
