# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer base"""
import torch


class Optimizer(torch.optim.Optimizer):
    r"""Base class for ratex optimizers

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def zero_grad(
        self, set_to_none: bool = False, inplace_update: bool = False
    ):  # pylint: disable=arguments-differ
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting gradients to zero, set_to_none gets rid of
                gradient tensors directly to have lower memory footprint and improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by
                a backward pass, ``.grad`` are guaranteed to be None for params that did not
                receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
            inplace_update (bool): enable gradient inplace update
        """
        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        if set_to_none:
                            param.grad = None
                        else:
                            if param.grad.grad_fn is not None:
                                param.grad.detach_()
                            else:
                                param.grad.requires_grad_(False)
                            if inplace_update:
                                # If new_grad is to be written in place to old_grad,
                                # old_grad must be an input of the relay function. Since
                                # old_grad is not an input for the operator old_grad.zero_,
                                # we cannot use it.
                                # In the long run, we may modify the zero_ LTC node and
                                # take the tensor as its input.
                                param.grad.sub_(param.grad)
                            else:
                                param.grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError
