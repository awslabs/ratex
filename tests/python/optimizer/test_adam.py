import pytest
import torch

from torch_mnm.optimizer import Adam
from torch_mnm.testing import TorchLeNet, fake_image_dataset, verify, train, with_seed


@pytest.mark.xfail(reason="Need to fix the accuracy issue")
@with_seed(0)
@pytest.mark.parametrize("device", ["cpu", "lazy"])
def test_adam(device):
    """Test Razor Adam implementation against PyTorch Adam.
    When device=cpu, we run Razor Adam as another PyTorch optimizer to make sure
    its functionality is the same as PyTorch Adam.
    """
    batch_size = 1
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    # FIXME: adam on lazy is wrong when trim=False
    results_0 = train(device, model, dataset, optimizer=Adam, batch_size=batch_size, trim=True)
    results_1 = train("cpu", model, dataset, optimizer=torch.optim.Adam, batch_size=batch_size)
    verify(results_0, results_1, tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
