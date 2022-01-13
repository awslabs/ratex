import functools

import torch

@functools.lru_cache(maxsize=None)
def tensor(scalar, dtype=torch.float32, device="xla"):
    return torch.tensor([scalar], dtype=dtype).to(device)


def tensor_like(scalar, t):
    return tensor(scalar, dtype=t.dtype, device=t.device)
