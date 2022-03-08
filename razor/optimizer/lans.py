# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-docstring, too-many-arguments, too-many-locals, too-many-branches
# pylint: disable=arguments-differ, unused-argument
import torch

from .optimizer import Optimizer


class LANS(Optimizer):
    """LANS optimizer."""

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        adam_w_mode=True,
        grad_averaging=True,
        set_grad_none=True,
        normalize_grad=True,
    ):
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            normalize_grad=normalize_grad,
        )
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        if not self.adam_w_mode:
            raise NotImplementedError("adam_w_mode == False is not implemented")

    def zero_grad(self, set_to_none: bool = False):
        if self.set_grad_none:
            for group in self.param_groups:
                for param in group["params"]:
                    param.grad = None
        else:
            super().zero_grad(set_to_none)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            grad_averaging = 1 if group["grad_averaging"] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            # create lists for multi-tensor apply
            g_16, _, p_16, m_16, v_16 = [], [], [], [], []
            g_32, _, p_32, m_32, v_32 = [], [], [], [], []

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedLANS does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                state = self.state[param]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(
                        param.data.size(), dtype=param.data.dtype, device=param.data.device
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg_sq"] = torch.zeros(
                        param.data.size(), dtype=param.data.dtype, device=param.data.device
                    )

                # Buffer for scaled grad
                # scaled_grad = torch.zeros_like(p.data)
                if param.dtype == torch.float16:
                    g_16.append(param.grad.data)
                    # q_16.append(scaled_grad)
                    p_16.append(param.data)
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])
                elif param.dtype == torch.float32:
                    g_32.append(param.grad.data)
                    # q_32.append(scaled_grad)
                    p_32.append(param.data)
                    m_32.append(state["exp_avg"])
                    v_32.append(state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedLANS only support fp16 and fp32.")

            if len(g_16) > 0:
                self._multi_tensor_lans(  # [g_16, q_16, p_16, m_16, v_16],
                    [g_16, p_16, m_16, v_16],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    bias_correction,
                    group["weight_decay"],
                    grad_averaging,
                    self.adam_w_mode,
                    group["normalize_grad"],
                )
            if len(g_32) > 0:
                self._multi_tensor_lans(  # [g_32, q_32, p_32, m_32, v_32],
                    [g_32, p_32, m_32, v_32],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    bias_correction,
                    group["weight_decay"],
                    grad_averaging,
                    self.adam_w_mode,
                    group["normalize_grad"],
                )

        return loss

    def _multi_tensor_lans(
        self,
        tensor_lists,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay,
        grad_averaging,
        mode,
        normalize_grad,
    ):
        for grad, param, momentum, variant in zip(*tensor_lists):
            self._lans(
                param,
                grad,
                momentum,
                variant,
                lr,
                beta1,
                beta2,
                epsilon,
                step,
                bias_correction,
                weight_decay,
                grad_averaging,
                mode,
                normalize_grad,
            )

    @staticmethod
    def _norm(x):
        return torch.sqrt(torch.sum(x * x))

    def _lans(
        self,
        param,
        grad,
        momentum,
        variant,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay,
        grad_averaging,
        mode,
        normalize_grad,
    ):
        p_norm = self._norm(param)
        scaled_p = param * weight_decay
        g_norm = self._norm(grad)
        scaled_g = grad
        if normalize_grad:
            scaled_g = torch.where(g_norm > 0, scaled_g / (g_norm + epsilon), scaled_g)
        beta3 = 1.0 - beta1 if grad_averaging else 1.0
        beta4 = 1.0 - beta2
        bias_correction1 = 1.0 - beta1 ** step if bias_correction else 1.0
        bias_correction2 = 1.0 - beta2 ** step if bias_correction else 1.0
        # Adam
        momentum.multiply_(beta1).add_(scaled_g * beta3)
        variant.multiply_(beta2).add_(scaled_g * scaled_g * beta4)
        # bias correction
        m_unbiased = momentum / torch.tensor(
            bias_correction1, dtype=momentum.dtype, device=momentum.device
        )
        v_unbiased = variant / torch.tensor(
            bias_correction2, dtype=variant.dtype, device=variant.device
        )
        # calculate updates
        denom = torch.sqrt(v_unbiased) + epsilon
        ratios1 = m_unbiased / denom
        tmp1 = ratios1 + scaled_p
        tmp1_norm = self._norm(tmp1)
        ratio1 = torch.where(
            torch.bitwise_and(p_norm > 0, tmp1_norm > 0),
            p_norm / tmp1_norm,
            torch.ones_like(p_norm),
        )
        param.subtract_(lr * beta1 * ratio1 * tmp1)
        ratios2 = scaled_g / denom
        tmp2 = ratios2 + scaled_p
        tmp2_norm = self._norm(tmp2)
        ratio2 = torch.where(
            torch.bitwise_and(p_norm > 0, tmp2_norm > 0),
            p_norm / tmp2_norm,
            torch.ones_like(p_norm),
        )
        param.subtract_(lr * beta3 * ratio2 * tmp2)
