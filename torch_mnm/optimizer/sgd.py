"""SGD optimizer"""
# pylint: disable=too-many-locals, invalid-name
import math

import torch

from mnm import distributed as dist


class SGD(torch.optim.Optimizer):
    """distributed SGD optimizer."""

    def __init__(self, params, lr=0.1, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        dctx = dist.get_context()
        self._zero_opt_level = dctx.zero_opt_level
        self._rank = dctx.rank
        self._world_size = dctx.size

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            momentum = None
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if momentum is None:
                        momentum = torch.FloatTensor([group["momentum"]]).to(p.device)

                    if not state:
                        if self._need_partition(p):
                            state["momentum_buffer"] = self._create_partitioned_buffer(p)
                        else:
                            state["momentum_buffer"] = torch.zeros_like(p, requires_grad=False)

                    momentum_buffer = state["momentum_buffer"]
                    if self._need_partition(p):
                        part_size = momentum_buffer.shape[0]

                        # Pad the tensor if it's not divisble by the number of ranks
                        # TODO: Instead of padding the entire tensor, we should pad only the last
                        # partitioned part to have better performance
                        if self._world_size * part_size > p.shape[0]:
                            # Padding the first dim. PyTorch pad parameter is in the reversed order.
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

                        # This line will be eventually replaced by allgather,
                        # so the index doesn't matter
                        p.scatter_(
                            dim=0,
                            index=torch.zeros_like(new_weight_slice, dtype=torch.int64),
                            src=new_weight_slice,
                        )
                    else:
                        momentum_buffer.mul_(momentum).add_(p.grad)
                        p.add_(momentum_buffer, alpha=-lr)

    def _need_partition(self, data):
        return self._zero_opt_level > 0 and data.shape[0] >= self._world_size

    def _create_partitioned_buffer(self, data):
        new_shape = list(data.shape)
        new_shape[0] = math.ceil(new_shape[0] / self._world_size)
        return torch.zeros(*new_shape, dtype=data.dtype, device=data.device, requires_grad=False)
