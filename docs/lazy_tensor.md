<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# What is Lazy Tensor and How it Works

In this document, we briefly introduce lazy tensor, the core idea used in this project that bridges the gap between PyTorch runtime interpreter and static compilation. We will train LeNet-5 as an end-to-end example to demonstrate hoe it works. After reading this document, you should be more conformable on writing a lazy-tensor-friendly PyTorch program, as well as attending to the Ratex development.

In PyTorch, a deep learning model is represented as a graph, where nodes are operations such as matrix multiplication, convolutions, and activation functions, and edges are control and data dependencies. When users execute the model, PyTorch runtime walks through the graph and executes each operators one-by-one. There are two major advantages of executing whatever operator it is visiting without caring about the whole graph are: 1) It is straightforward for PyTorch to offer eager mode for better user experience. 2) It is easy to support dynamic shapes and control flow.

On the other hand, the dynamic graph interpreter misses the opportunities of graph-level optimizations, such as operator fusion and expression simplification. As a result, [Lazy Tensor Core](https://arxiv.org/pdf/2102.13267.pdf) is presented to bridge this gap. In short, lazy tensor allows users to still write PyTorch programs, but is able to capture an entire graph for optimization before execution. Next, we use a simple example to explain how it works.

We first implement a simple model (e.g., LeNet-5) in PyTorch and train it on CPU:

```python
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

# Implement LeNet-5
class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super().__init__()
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
        out = torch.relu(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

# Prepare a fake dataset and dataloader.
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
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=1, shuffle=False, num_workers=1
    )
    for x in ["train", "val"]
}

# Configurations
device = "cpu"
num_epochs = 10

# Train the model with SGD optimizer on CPU.
model = TorchLeNet()
model.to(device)
model.train()

criterion = lambda pred, true: nn.functional.nll_loss(nn.LogSoftmax(dim=-1)(pred), true)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    running_loss = 0.0

    # Iterate over data.
    for inputs, labels in dataloaders["train"]:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_sizes["train"]
    epoch_acc = 0
    print("Epoch {}/{} loss: {:.4f}".format(epoch, num_epochs - 1, epoch_loss))
```

Output:

```
python3 demo.py
Epoch 0/9 loss: 2.3010
Epoch 1/9 loss: 2.2986
Epoch 2/9 loss: 2.2962
Epoch 3/9 loss: 2.2938
Epoch 4/9 loss: 2.2914
Epoch 5/9 loss: 2.2890
Epoch 6/9 loss: 2.2866
Epoch 7/9 loss: 2.2842
Epoch 8/9 loss: 2.2818
Epoch 9/9 loss: 2.2795
```

Now, we slightly modify this script to make use of Ratex and GPU. We first import Ratex and make this model "lazy".

```python
# Configurations
device = lm.lazy_device()  # Point 1
num_epochs = 10

# Train the model with SGD optimizer.
model = TorchLeNet()
model.train()

criterion = lambda pred, true: nn.functional.nll_loss(nn.LogSoftmax(dim=-1)(pred), true)

model = ratex.jit.script(model) # Point 2
model.to(device) # Point 3
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    running_loss = 0.0

    # Iterate over data.
    for inputs, labels in dataloaders["train"]:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True
        # Point 4 start
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Point 4 end
        lm.mark_step() # Point 5
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_sizes["train"]
    epoch_acc = 0
    print("Epoch {}/{} loss: {:.4f}".format(epoch, num_epochs - 1, epoch_loss))
```

Outputs:

```
ENABLE_PARAM_ALIASING=true RATEX_DEVICE=GPU python3 demo.py
Epoch 1/9 loss: 2.3147
Epoch 2/9 loss: 2.3128
Epoch 3/9 loss: 2.3110
Epoch 4/9 loss: 2.3091
Epoch 5/9 loss: 2.3073
Epoch 6/9 loss: 2.3054
Epoch 7/9 loss: 2.3036
Epoch 8/9 loss: 2.3017
Epoch 9/9 loss: 2.2999
```

We now explain what happens with the points marked in the above script.

**Point 1**: We use `lm.lazy_device()` instead of `lazy` to change the default device of PyTorch. It's useful to make sure your program run correctly when you use a device with device index other than 0.

**Point 2**: We use a Ratex specific API to wrap a PyTorch model. What happens here is that we trace the forward graph with TorchScript to capture the whole forward model for two purposes. First, capturing control flows. Second, performing auto-differentiation in Ratex instead of in PyTorch.

**Point 3**: Instead of `.to("cpu")` or `.to("cuda")`, we now put the model on **lazy** device. This is a virtual device in PyTorch since v1.11, and Ratex registers itself as the backend implementation of "lazy" device. In other words, when a PyTorch model is on the lazy device, it will be powered by Ratex. Since this registration happens when importing Ratex, transferring the mode to lazy device without importing Ratex results in the following error:

```
NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'Lazy' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'aten::empty.memory_format' is only available for these backends: [Dense, Conjugate, VmapMode, FuncTorchGradWrapper, MLC, VE, Lazy, PrivateUse1, PrivateUse2, PrivateUse3, ...].
```

**Point 4**: After the model has been placed on lazy device and wrapped by Ratex, when PyTorch executing these lines (forward inference, loss calculation, backward propagation, parameter updating), NO actual computation is performed yet. Instead, every operator in Ratex (i.e., lazy tensor core) produces a lazy tensor, which only includes the tensor information (e.g., shape and data type) and how it was generated (by which operator and which input tensors). By tracing these lazy tensors, we construct a static computation graph and defer their execution until Point 4. As a result, if you time these lines in the code, you will find that they only need a few milliseconds whatever the model size is.

**Point 5**: This is the most important part in Ratex. As mentioned in the previous point, the computation of the model on lazy device will be deferred, and this particular API (i.e., `mark_step`) is the one that informs Ratex to compile and execute all deferred computations so far. As a result, if you time this line, you will find that it takes a relative long time especially on the first iteration.

Note that in addition to the `mark_step` API, other manipulations that attempt to retrieve tensor values will also invoke `mark_step`. For example, if you `print(loss.item())` even before calling `loss.backward()`, then the forward graph will be compiled and executed at that line in order to materialize the real value of the `loss` tensor. Consequently, it is important to never or less frequently materialize tensor values during the entire training process. Otherwise the computations between two `mark_step`s will be compiled as separate graphs, which not only prevents global graph-level optimizations from happening, but also introduces more compilation and data processing overheads.

Finally, we use a timer to illustrate our statements above. We profiled the first 3 epochs for both PyTorch on CPU and Ratex on GPU. The numbers are in milliseconds. Note that we are not comparing their performance in this experiment.

Mini-batch 1 | PyTorch | Ratex
-- | -- | --
outputs = model(inputs) | 7.69 | 1414.14
loss = criterion(outputs,   labels) | 1.07 | 0.34
loss.backward() | 3.88 | 0.41
optimizer.step() | 1.73 | 0.76
lm.mark_step() | N/A | 967.36

Mini-batch 2 | PyTorch | Ratex
-- | -- | --
outputs = model(inputs) | 2.08 | 1.01
loss = criterion(outputs,   labels) | 0.96 | 0.48
loss.backward() | 1.43 | 0.69
optimizer.step() | 1.83 | 0.55
lm.mark_step() | N/A | 497.85

Mini-batch 3 | PyTorch | Ratex
-- | -- | --
outputs = model(inputs) | 6.37 | 0.97
loss = criterion(outputs,   labels) | 0.93 | 0.45
loss.backward() | 1.85 | 0.43
optimizer.step() | 1.78 | 0.61
lm.mark_step() | N/A | 0.51

We highlight some points in the above table. Note that since we configure each epoch to have only one mini-batch, "epoch" is equivalent to "mini-batch" or "iteration" in the following descriptions.

1. In epoch 1, we can see that Ratex takes 1.4 seconds in forward. This is because Ratex is converting the PyTorch model to Ratex, as mentioned in **Point 2**. We could notice that this conversion only needs to be done once, so the time in rest epochs are much shorter.
2. In epoch 1, the `mark_step` in Ratex takes almost 1 second, because it includes 1) model graph optimization, 2) kernel code generation and compilation, 3) model execution. Again, the graph optimization and kernel compilation only need to be done just once, so we can see a much shorter time in the rest epochs.
3. In all epochs, backward propagation, loss calculation, and parameter updating take less than 1 millisecond in Ratex. This is because there is no computation happening in these lines. Instead, they only produces lazy tensor for graph construction.
4. When benchmarking Ratex, it is inaccurate to simply sum up all execution time of these lines, because the tracing latency (all lines shown in this table except for the `mark_step`) can be hidden. We will explain this in detail in another article.
