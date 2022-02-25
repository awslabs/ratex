# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer utilities."""
import functools

import torch


@functools.lru_cache(maxsize=None)
def tensor(scalar, dtype=torch.float32, device="lazy"):
    """Create a tensor with a single value."""
    return torch.tensor([scalar], dtype=dtype).to(device)


def tensor_like(scalar, like_tensor):
    """Create a tensor with a single value with the same dtype and device as the reference one."""
    return tensor(scalar, dtype=like_tensor.dtype, device=like_tensor.device)
