"""Optimizer base"""
import torch
from mnm import distributed as dist

from . import _functional as F
from . import utils

class Optimizer(torch.optim.Optimizer):
    r"""Base class for razor optimizers

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    def zero_grad(self, set_to_none: bool = False, inplace_update: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
            inplace_update (bool): enable gradient inplace update
        """
        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if inplace_update:
                                # If new_grad is to be written in place to old_grad,
                                # old_grad must be an input of the relay function. Since
                                # old_grad is not an input for the operator old_grad.zero_,
                                # we cannot use it.
                                # In the long run, we may modify the zero_ LTC node and
                                # take the tensor as its input.
                                p.grad.sub_(p.grad)
                            else:
                                p.grad.zero_()
