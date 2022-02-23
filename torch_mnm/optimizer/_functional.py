"""The functional implementation of optimizers."""
# pylint: disable=too-many-arguments, too-many-locals
from typing import List
import math

import torch
from torch import Tensor

from . import utils

def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    assert len(params) == 1

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(utils.tensor_like(beta1, exp_avg))
        exp_avg.add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(utils.tensor_like(beta2, exp_avg_sq)).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() /
                utils.tensor_like(math.sqrt(bias_correction2), max_exp_avg_sqs[i])).add_(
                utils.tensor_like(eps, max_exp_avg_sqs[i]))
        else:
            denom = (exp_avg_sq.sqrt()
                / utils.tensor_like(math.sqrt(bias_correction2), exp_avg_sq)).add_(
                utils.tensor_like(eps, exp_avg_sq))

        step_size = lr / bias_correction1

        # FIXME: neuron errors if inplace update
        # param.addcdiv_(exp_avg, denom, value=-step_size)
        return torch.addcdiv(param, exp_avg, denom, value=-step_size)
