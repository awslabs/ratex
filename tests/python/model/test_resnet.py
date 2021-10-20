import copy

import torch
import torch_mnm
import torch_mnm.utils
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as lm


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy

class FakeData(Dataset):
    def __init__(self, size, channel, height, width, transform=None, target_transform=None):
        self.x = torch.randn(size, channel, height, width)
        self.y = np.random.randint(0, vocab_size, size=(size, seq_length))
        self.y = np.eye(vocab_size, dtype="float32")[self.y]
        self.y = torch.tensor(self.y)
        self.size = size
        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


def train(device, model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=1)
    dataset_size = len(dataset)
    model = model.to(device, dtype=torch.float32)
    model.train()
    # adapting loss cacluation from 
    # https://www.programmersought.com/article/86167037001/
    # this doesn't match nn.NLLLoss() exactly, but close...
    criterion = lambda pred, true: -torch.sum(pred * true) / true.size(0)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 10
    if device == "xla":
        model = torch_mnm.jit.script(model)
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels_one_hot = torch.tensor(np.eye(10, dtype=np.float32)[labels])
            labels_one_hot = labels_one_hot.to(device) # One-hot
            optimizer.zero_grad()
            outputs = model(inputs)[0]
            # log_softmax diverges a lot from CPU in QEMU
            outputs = torch.nn.functional.log_softmax(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lm.mark_step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = 0
        print('{} Loss: {:.4f}'.format(
              "train", epoch_loss))


model_mnm = torchvision.models.resnet18()
model_cpu = copy.deepcopy(model_mnm)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
}
image_datasets = datasets.FakeData(size=1, image_size=(3, 224, 224), num_classes=100,
                                   transform=transforms.ToTensor())
print("mnm starts...")
train("xla", model_mnm, image_datasets)
print("cpu starts...")
train("cpu", model_cpu, image_datasets)
