# pylint: disable=unused-variable
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_mnm.optimizer import Adam
from torch_mnm.testing import TorchLeNet, fake_image_dataset, verify, train, with_seed

@with_seed(0)
@pytest.mark.parametrize("device", ["cpu", "xla"])
def test_adam(device):
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    # FIXME: adam on xla is wrong when trim=False
    results_0 = train(device, model, dataset, optimizer=Adam, batch_size=batch_size, trim=True)
    results_1 = train("cpu", model, dataset, optimizer=torch.optim.Adam, batch_size=batch_size)
    verify(results_0, results_1, tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
