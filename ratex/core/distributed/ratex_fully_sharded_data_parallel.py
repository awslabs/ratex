import torch.nn as nn
import torch.nn.functional as F

import ratex.optimizer as optim
from ratex.core.lazy_model import all_gather, all_reduce, REDUCE_SUM

from raf import distributed as dist

class RatexFullyShardedDataParallel(nn.Module):
    def __init__(self, module, optimizer=optim.SGD, optimizer_config={}):
        super().__init__()
        comm = dist.get_communicator()
        self.rank = comm.rank
        self.world_size = comm.size
        self.module = module
        self.params = list(self.module.parameters())

        # Shard parameters for use in optimizer
        self._shard_parameters()
        # Optimizer initialization
        self.optimizer = optimizer(self.parameters(shards=True), **optimizer_config)

    def _shard_parameters(self):
        """
        Create list of this ranks shards
        Shards are stored in a tuple with the respective parameter ie (parameter, shard)
        """
        self.sharded_params = []
        for p in self.params:
            shard_data = self._get_shard(p.data)
            p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
            self.sharded_params.append((p, p_shard))

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

    def parameters(self, shards=False):
        """
        Generator for parameters
        If shards is set to True, then the respective parameter shards of this rank are yielded instead
        """
        if not shards:
            yield from self.params
        elif hasattr(self, "sharded_params"):
            for p, s in self.sharded_params:
                yield s

    def forward(self, *args, **kwargs):
        """
        Calculate the output of the model using the wrapped module
        """
        outputs = self.module(*args, **kwargs)
        return outputs

    def zero_grad(self, *args, **kwargs):
        """
        Zero gradients of the parameters
        """
        # Todo: Calling self.optimizer.zero_grad() may not be necessary since the shard gradients are replaced
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Step the optimizer and update parameter weights
        """
        # Reduce full gradients across ranks and assign gradient shards to the respective parameter shards
        for param, shard in self.sharded_params:
            if param.grad is not None:
                all_reduce(REDUCE_SUM, [param.grad], scale=1.0 / self.world_size)
                shard.grad = self._get_shard(param.grad)

        # Step the wrapped optimizer
        loss = self.optimizer.step(*args, **kwargs)

        # All gather the resulting new weights across the ranks and assign them to the full parameters
        for param, shard in self.sharded_params:
            all_gather(shard.data, dim=0, output=param.data)

        return loss
