# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module wrapper implementing ZeRO-1 in an FSDP style interface"""
import torch.nn as nn
import torch.nn.functional as F

from raf import distributed as dist

import ratex.optimizer as optim
from ratex.core.lazy_model import all_gather, all_reduce, REDUCE_SUM


class RatexFullyShardedDataParallel(nn.Module):
    """FSDP ZeRO-1 wrapper"""
    def __init__(self, module, optimizer=optim.SGD, optimizer_config=None):
        super().__init__()
        comm = dist.get_communicator()
        self.rank = comm.rank
        self.world_size = comm.size
        self.module = module
        self.params = list(self.module.parameters())

        # Shard parameters for use in optimizer
        self._shard_parameters()
        # Optimizer initialization
        self.optimizer = optimizer(self.sharded_parameters(), **optimizer_config or {})

    def _shard_parameters(self):
        """
        Create list of this ranks shards
        Shards are stored in a tuple with the respective parameter ie (parameter, shard)
        """
        self.sharded_params = []
        for param in self.params:
            shard_data = self._get_shard(param.data)
            shard = nn.Parameter(shard_data, requires_grad=param.requires_grad)
            self.sharded_params.append(shard)

    def _get_shard(self, tensor):
        """
        Get the shard of the input tensor that is associated with this rank
        The input tensor is padded if the length is not divisible across the ranks
        """
        if tensor.numel() % self.world_size != 0:
            pad_size = self.world_size - (tensor.numel() % self.world_size)
            tensor = F.pad(tensor, (0, pad_size))
        tensor = tensor.chunk(self.world_size)[self.rank]
        return tensor

    def sharded_parameters(self):
        """
        Generator for the sharded parameters
        """
        yield from self.sharded_params

    def forward(self, *args, **kwargs):
        """
        Calculate the output of the model using the wrapped module
        """
        outputs = self.module(*args, **kwargs)
        return outputs

    def step(self, *args, **kwargs):
        """
        Step the optimizer and update parameter weights
        """
        # Reduce full gradients across ranks
        # Assign gradient shards to the respective parameter shards
        for param_index in range(len(self.params)):
            param, shard = self.params[param_index], self.sharded_params[param_index]
            if param.grad is not None:
                all_reduce(REDUCE_SUM, [param.grad], scale=1.0 / self.world_size)
                shard.grad = self._get_shard(param.grad)

        # Step the wrapped optimizer
        loss = self.optimizer.step(*args, **kwargs)

        # All gather the new weights across the ranks and assign them to the full parameters
        for param_index in range(len(self.params)):
            param, shard = self.params[param_index], self.sharded_params[param_index]
            all_gather(shard.data, dim=0, output=param.data)

        return loss
