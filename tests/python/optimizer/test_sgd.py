import pytest
import torch
from torch_mnm.optimizer import SGD
from torch_mnm.testing import TorchLeNet, fake_image_dataset, train, verify, with_seed


@with_seed(0)
@pytest.mark.parametrize("momentum", [0.0, 0.1])
def test_sgd(momentum):
    """Test Razor SGD implementation against PyTorch SGD."""
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
