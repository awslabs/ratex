# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adam optimizer"""
# pylint: disable=too-many-arguments, too-many-locals
import math
from importlib import import_module

import torch
from raf import distributed as dist

from ratex.core.lazy_model import all_gather

from .optimizer import Optimizer


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        mark_step (boolean, optional): whether to mark step after each parameter
            update (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        mark_step=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        # TODO(@hzfan): support amsgrad
        if amsgrad is not False:
            raise NotImplementedError("amsgrad=True is not yet supported")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        # Distributed configs
        dcfg = dist.get_config()
        comm = dist.get_communicator()
        self._zero_opt_level = dcfg.zero_opt_level
        self._rank = comm.rank
        self._world_size = comm.size
        self._lm = import_module("ratex.lazy_tensor_core.core.lazy_model") if mark_step else None

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        # pylint: disable=too-many-branches, too-many-statements
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for param in group["params"]:
                if param.grad is not None:
                    param_with_grad_global = param
                    if param.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, "
                            "please consider SparseAdam instead"
                        )

                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # pylint: disable=line-too-long
                        # FIXME: lowering zeros_like to ltc triggers compile error
                        # state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # pylint: enable=line-too-long
                        if self._need_partition(param):
                            state["exp_avg"] = self._create_partitioned_buffer(
                                param, dtype=torch.float32
                            )
                            state["exp_avg_sq"] = self._create_partitioned_buffer(
                                param, dtype=torch.float32
                            )
                        else:
                            state["exp_avg"] = torch.zeros(
                                param.data.size(), dtype=torch.float32
                            ).to(device=param.data.device)
                            state["exp_avg_sq"] = torch.zeros(
                                param.data.size(), dtype=torch.float32
                            ).to(device=param.data.device)
                        if param.dtype in (torch.float16, torch.bfloat16):
                            # master weight param
                            state["param"] = self._partition(param.data, state["exp_avg"]).float()
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    assert exp_avg.shape == exp_avg_sq.shape
                    param_with_grad_local = (
                        state["param"] if "param" in state else self._partition(param.data, exp_avg)
                    )
                    grad = self._partition(param.grad, exp_avg)
                    if grad.dtype in (torch.float16, torch.bfloat16):
                        grad = grad.float()
                    state["step"] += 1
                    state_step = state["step"]

                    bias_correction1 = 1 - beta1**state_step
                    bias_correction2 = 1 - beta2**state_step

                    if group["weight_decay"] != 0:
                        grad = grad.add(param_with_grad_local, alpha=group["weight_decay"])

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if group["amsgrad"]:
                        raise NotImplementedError("amsgrad==True is not yet supported")
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(group["eps"])

                    step_size = group["lr"] / bias_correction1
                    param_with_grad_local.addcdiv_(exp_avg, denom, value=-step_size)
                    if param.dtype == torch.float16:
                        updated_param_with_grad_local = param_with_grad_local.half()
                    elif param.dtype == torch.bfloat16:
                        updated_param_with_grad_local = param_with_grad_local.bfloat16()
                    else:
                        updated_param_with_grad_local = param_with_grad_local

                    if self._need_partition(param_with_grad_global):
                        all_gather(
                            updated_param_with_grad_local, dim=0, output=param_with_grad_global
                        )
                    elif param.dtype in (torch.float16, torch.bfloat16):
                        param_with_grad_global.copy_(updated_param_with_grad_local)

                    if self._lm:
                        updated_param_with_grad_local = None
                        grad = None
                        self._lm.mark_step()
        return loss

    def _need_partition(self, data):
        return self._zero_opt_level > 0 and data.shape[0] >= self._world_size

    def _create_partitioned_buffer(self, data, dtype=None):
        new_shape = list(data.shape)
        new_shape[0] = math.ceil(new_shape[0] / self._world_size)
        return torch.zeros(
            *new_shape, dtype=data.dtype if dtype is None else dtype, requires_grad=False
        ).to(device=data.device)

    def _partition(self, global_data, local_data):
        if self._need_partition(global_data):
            part_size = local_data.shape[0]
            # Pad the tensor if it's not divisble by the number of ranks
            # TODO: Instead of padding the entire tensor, we should pad only the last
            # partitioned part to have better performance
            if self._world_size * part_size > global_data.shape[0]:
                pad_width = [0 for _ in range(len(global_data.shape) * 2)]
                pad_width[-1] = self._world_size * part_size - global_data.shape[0]
                part_data = torch.nn.functional.pad(global_data, pad_width)
            else:
                part_data = global_data
            part_data = part_data[self._rank * part_size : (self._rank + 1) * part_size]
        else:
            part_data = global_data
        return part_data
