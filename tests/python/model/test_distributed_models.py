# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test torchvision models."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import torch
import torch.optim as optim
import torchvision
from raf.testing import get_dist_comm_info, skip_dist_test
from raf import distributed as dist

from ratex.lazy_tensor_core.core.lazy_model import lazy_device
from ratex.optimizer import LANS, SGD, Adam
from ratex.testing import TorchLeNet, fake_image_dataset, train
from ratex.testing import (
    with_seed,
    with_enable_param_aliasing,
)

SKIP_REASON = "Distribution is not enabled or #rank is not expected"


@with_seed(0)
@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.skip("Seeing issues with CUDDN, need to be fixed")
@with_enable_param_aliasing
@pytest.mark.parametrize(
    "optimizer", [(SGD, {"lr": 0.001, "momentum": 0.1}, 1), (Adam, {"lr": 0.001}, 2)]
)
@pytest.mark.parametrize("grad_inplace", [True, False])
@pytest.mark.parametrize("enable_zero1", [False])
def test_lenet_distributed(optimizer, grad_inplace, enable_zero1):
    total_rank, rank, local_rank = get_dist_comm_info()
    micro_batch_size = 4
    acc_grad_steps = 1
    batch_size = micro_batch_size * acc_grad_steps
    dataset = fake_image_dataset(batch_size, 1, 28, 10)
    model = TorchLeNet()
    device = lazy_device(local_rank)
    dtype = torch.float32

    if enable_zero1:
        dcfg = dist.get_config()
        dcfg.zero_opt_level = 1
        dtype = torch.float16

    lazy_losses = train(
        str(device),
        model,
        dataset,
        dtype=dtype,
        optimizer=optimizer[0],
        optimizer_params=optimizer[1],
        batch_size=micro_batch_size,
        acc_grad_steps=acc_grad_steps,
        num_epochs=3,
        trim=True,
        set_to_none=not grad_inplace,
        reduce_gradients=True,
    )


if __name__ == "__main__":
    if os.environ.get("RAF_FILE_STORE_PATH", None):
        dist.set_default_communicator("void")
        comm = dist.get_communicator()
        size = int(
            os.environ.get("OMPI_COMM_WORLD_SIZE", None) or os.environ.get("MPIRUN_NPROCS", None)
        )
        rank = int(
            os.environ.get("OMPI_COMM_WORLD_RANK", None) or os.environ.get("MPIRUN_RANK", None)
        )
        comm.size = size
        comm.rank = rank
        comm.local_size = size
        comm.local_rank = rank
    exit_code = pytest.main([__file__])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
