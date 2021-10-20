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
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
