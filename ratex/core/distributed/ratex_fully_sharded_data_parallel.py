# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module wrapper implementing ZeRO-1 in an FSDP style interface"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from raf import distributed as dist

import ratex.optimizer as optim
from ratex.core.lazy_model import all_gather, all_reduce, REDUCE_SUM


class RatexFullyShardedDataParallel(nn.Module):
    r"""
    FSDP ZeRO-1 wrapper
    Args:
      module (nn.Module): The module to be wrapped and sharded
      optimizer (torch.optim.Optimizer): The constructor to be used for initializing the optimizer
        Default: ratex.Optimizer.SGD
      optimizer_config (dict): Arguments to be passed into the optimizer constructor
        Default: None - Corresponds to an empty dictionary
    """

    def __init__(
        self,
        module: nn.Module,
        optimizer: torch.optim.Optimizer = optim.SGD,
        optimizer_config: dict = None,
    ):
        super().__init__()
        comm = dist.get_communicator()
        self.rank = comm.rank
        self.world_size = comm.size
        self.module = module
        self.params = list(self.module.parameters())

        # Shard parameters for use in optimizer
        self.sharded_params = []
        self._shard_parameters()
        # Optimizer initialization
        self.optimizer = optimizer(self.sharded_parameters(), **optimizer_config or {})

    def _shard_parameters(self):
        """
        Create a list of this ranks shards
        Shards are stored in a tuple with the respective parameter ie (parameter, shard)
        Args: None
        Returns: None
        """
        for param in self.params:
            shard_data = self._shard_tensor(param.data)
            shard = nn.Parameter(shard_data, requires_grad=param.requires_grad)
            self.sharded_params.append(shard)

    def _shard_tensor(self, tensor: torch.Tensor):
        """
        Get the shard of the input tensor that is associated with this rank
        The input tensor is padded if the length is not divisible across the ranks
        Args:
          tensor (torch.Tensor): tensor to be sharded
        Returns:
          A tensor that corresponds to the respective shard of this rank
        """
        if self.rank == self.world_size - 1 and tensor.size()[0] % self.world_size != 0:
            padding = [0] * (len(tensor.size()) * 2)
            padding[-1] = self.world_size - (tensor.size()[0] % self.world_size)
            tensor = F.pad(tensor, tuple(padding))
        tensor = tensor.chunk(self.world_size)[self.rank]
        return tensor

    def sharded_parameters(self):
        """
        Generator for the sharded parameters
        Args: None
        Yields: The parameter shards that correspond to this rank
        """
        yield from self.sharded_params

    def forward(self, *args, **kwargs):
        """
        Calculate the output of the model using the wrapped module
        Args:
          **args, **kwargs to be passed into the wrapped module's forward pass method
        Returns:
          The model forward pass output given the inputs
        """
        return self.module(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Step the optimizer and update parameter weights
        Args:
          **args, **kwargs to be passed into the optimizer's step method
        Returns:
          The calculated loss from the backwards pass
        """
        # Reduce full gradients across ranks
        # Assign gradient shards to the respective parameter shards
        for param, shard in zip(self.params, self.sharded_params):
            if param.grad is not None:
                all_reduce(REDUCE_SUM, [param.grad], scale=1.0 / self.world_size)
                shard.grad = self._shard_tensor(param.grad)

        # Step the wrapped optimizer
        loss = self.optimizer.step(*args, **kwargs)

        # All gather the new weights across the ranks and assign them to the full parameters
        for param, shard in zip(self.params, self.sharded_params):
            param.data = all_gather(shard.data, dim=0)[: param.data.shape[0]]

        return loss
