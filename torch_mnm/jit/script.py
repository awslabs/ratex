import copy

import torch

from .. import _TORCHMNMC
from .._lib import mnm
from ..value import ValueToHandle
from mnm._ffi.pass_ import AutoDiff, InferType
from mnm._core.module import IRModule


def to_torch_name(name):
    if name.startswith("model_"):
        assert name.startswith("model_")
        name = name[6:]
        name = name.replace("_", ".")
    return name


def to_mnm_name(name):
    return "model_" + name.replace(".", "_")


def get_positional_args(func, *args, **kwargs):
    """convert a mixture of positional args and keyword args to positional args only

    Parameters
    ----------
    func : relay.Function
    
    args:
        positional args

     kwargs:
        keyword args

    Returns
    -------
    ret:
        a list of positional args
    """
    ret = []
    i = 0
    mnm_kwargs = {to_mnm_name(k): v for k, v in kwargs.items()}
    for var in func.params:
        if var.name_hint in mnm_kwargs:
            param = mnm_kwargs[var.name_hint]
        else:
            param = args[i]
            i = i + 1
        assert isinstance(param, (torch.Tensor, torch.nn.parameter.Parameter))
        ret.append(param)
    assert i == len(args)
    return ret


class RelayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, *args):
        mod = IRModule.from_expr(func)
        mod = AutoDiff([])(InferType()(mod))
        func = mod["main"]
        handle = ValueToHandle(mnm._core.value.ClosureValue({}, func))
        
        func = _TORCHMNMC._mnm_to_tensor(handle)
        result = _TORCHMNMC._mnm_invoke_relay(func, args)
        ctx.bwd = result[1]
        return result[0]

    @staticmethod
    def backward(ctx, grad_output):
        ret = _TORCHMNMC._mnm_invoke_relay(ctx.bwd, [grad_output])
        ret = tuple([None] + ret)
        return ret


def asnumpy(x):
    """Helper function to convert x to numpy"""
    assert isinstance(x, torch.Tensor), f"{type(x)} is not supported"
    return x.cpu().detach().numpy()


def script(module: torch.nn.Module):
    """wraps a torch.nn.Module to support control flow
    
    Parameters
    ----------
    module : torch.nn.Module
    
    Returns
    -------
    ret: torch.device
        The mnm device
    """
    def f(*args, **kwargs):
        # TODO: cache the result
        # TODO: eliminate shape_dict
        # TODO: use torch.jit.script
        assert len(args) == 1
        assert not kwargs
        shape_dict = {
            "input0": ((list(args[0].shape), str(args[0].dtype).split(".")[-1]))
        }
        traced_module = copy.deepcopy(module)
        model = mnm.frontend.from_pytorch(traced_module, shape_dict)
        record = model._internal(mnm.array(asnumpy(args[0])))
        positional_args = get_positional_args(record.mod["main"], *args, **dict(module.named_parameters()))
        return RelayFunction.apply(record.mod["main"], *positional_args)

    return f
