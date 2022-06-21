# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from ratex.optimizer import Adam
from ratex.testing import TorchLeNet, fake_image_dataset, verify, train, with_seed


@with_seed(0)
@pytest.mark.parametrize("device", ["cpu", "lazy"])
# FIXME: fail when trim is False
@pytest.mark.parametrize("trim", [True])
def test_adam(device, trim):
    """Test Ratex Adam implementation against PyTorch Adam.
    When device=cpu, we run Ratex Adam as another PyTorch optimizer to make sure
    its functionality is the same as PyTorch Adam.
    """
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    results_0 = train(device, model, dataset, optimizer=Adam, batch_size=batch_size, trim=trim)
    results_1 = train("cpu", model, dataset, optimizer=torch.optim.Adam, batch_size=batch_size)
    verify(results_0, results_1, tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
