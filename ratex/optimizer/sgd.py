# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGD optimizer"""
# pylint: disable=too-many-locals, invalid-name, too-many-branches
import math
from importlib import import_module

import numpy as np
import torch

from raf import distributed as dist
from ratex.core.lazy_model import all_gather

from .optimizer import Optimizer


class SGD(Optimizer):
    """distributed SGD optimizer."""

    def __init__(self, params, lr=0.1, momentum=0, mark_step=False):
        defaults = dict(lr=lr, momentum=momentum)
        dcfg = dist.get_config()
        comm = dist.get_communicator()
        self._zero_opt_level = dcfg.zero_opt_level
        self._rank = comm.rank
        self._world_size = comm.size
        self._lm = import_module("lazy_tensor_core.core.lazy_model") if mark_step else None

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if group["momentum"] == 0:
                        p.add_(p.grad, alpha=-lr)
                        if self._lm:
                            self._lm.mark_step()
                        continue

                    if not state:
                        if self._need_partition(p):
                            state["momentum_buffer"] = self._create_partitioned_buffer(p)
                        else:
                            state["momentum_buffer"] = torch.clone(p).detach().to(p.device)

                    momentum_buffer = state["momentum_buffer"]
                    if self._need_partition(p):
                        part_size = momentum_buffer.shape[0]

                        # Pad the tensor if it's not divisble by the number of ranks
                        # TODO: Instead of padding the entire tensor, we should pad only
                        # the last partitioned part to have better performance
                        if self._world_size * part_size > p.shape[0]:
                            # Padding the first dim. PyTorch pad parameter is in the
                            # reversed order.
                            pad_width = [0 for _ in range(len(p.shape) * 2)]
                            pad_width[-1] = self._world_size * part_size - p.shape[0]
                            grad = torch.nn.functional.pad(p.grad, pad_width)
                            data = torch.nn.functional.pad(p.data, pad_width)
                        else:
                            data = p.data
                            grad = p.grad

                        grad_slice = grad[self._rank * part_size : (self._rank + 1) * part_size]
                        data_slice = data[self._rank * part_size : (self._rank + 1) * part_size]

                        momentum_buffer.mul_(momentum).add_(grad_slice)

                        new_weight_slice = data_slice + momentum_buffer * (-lr)
                        all_gather(new_weight_slice, dim=0, output=p.data)

                        # Hints for LTC to not output the intermediate tensors
                        del grad_slice
                        del data_slice
                        del new_weight_slice
                    else:
                        momentum_buffer.mul_(momentum).add_(p.grad)
                        p.add_(momentum_buffer, alpha=-lr)
                    if self._lm:
                        self._lm.mark_step()

        return loss

    def _need_partition(self, data):
        return self._zero_opt_level > 0 and data.shape[0] >= self._world_size

    def _create_partitioned_buffer(self, data):
        part_size = math.ceil(data.shape[0] / self._world_size)
        initial_value = torch.clone(data).detach().cpu().numpy()
        pad_width = self._world_size * part_size - data.shape[0]

        if self._world_size * part_size > data.shape[0]:
            initial_value = np.pad(initial_value, (0, pad_width))

        initial_value_slice = initial_value[self._rank * part_size : (self._rank + 1) * part_size]
        return torch.from_numpy(initial_value_slice).to(data.device)
