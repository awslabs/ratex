# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ratex.optimizer import SGD
from ratex.testing import TorchLeNet, fake_image_dataset, train, verify, with_seed


@with_seed(0)
@pytest.mark.parametrize("momentum", [0.0, 0.1])
def test_sgd(momentum):
    """Test Ratex SGD implementation against PyTorch SGD."""
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    optimizer_params = {"lr": 0.01, "momentum": momentum}
    lazy_results = train(
        "lazy",
        model,
        dataset,
        optimizer=SGD,
        optimizer_params=optimizer_params,
        batch_size=batch_size,
        trim=True,
    )
    cpu_results = train(
        "cpu",
        model,
        dataset,
        optimizer=torch.optim.SGD,
        optimizer_params=optimizer_params,
        batch_size=batch_size,
    )
    verify(lazy_results, cpu_results, tol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
