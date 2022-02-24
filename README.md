# PyTorch/RAF

![CI-Lass-Pass](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/aire-meta-bot/5e763b919e5c0b91ceede34877869ec0/raw/razor-ci-badge-last-pass.json)

It aims to bridge torch models and RAF backends as follows. This repo is a POC.

```
from torch_mnm import TorchModel, mnm_device, mark_step
from torch import optim, nn
import torchvision

device = mnm_device()
model = getattr(torchvision.models, "some_model")()
# if control flow needs to be expressed in graph IR, uncomment the following line
# model = TorchModel(model)
model = model.train().to(device) 
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for data, target in train_loader:
  optimizer.zero_grad()
  data = data.to(device)
  target = target.to(device)
  output = model(data)  # type(output) == torch.Tensor
  loss = loss_fn(output, target)  # type(loss) == torch.Tensor
  loss.backward()
  optimizer.step()
  print(f"loss = {loss}") # trigger evaluation
  mark_step()  # trigger evaluation
```
