"""The JIT script."""
# pylint: disable=c-extension-no-member, protected-access
import copy

import torch
import mnm
from mnm._core.module import IRModule
from mnm._ffi.pass_ import AutoDiff, DeadCodeElimination, InferType

from .. import _TORCHMNMC
from .._lib import mnm
from ..value import ValueToHandle

_APIS = mnm._lib._get_apis()
InplaceUpdateAnalysis = _APIS.get("mnm.pass_.InplaceUpdateAnalysis", None)
CanonicalizeParamsForRAZOR = _APIS.get("mnm.pass_.CanonicalizeParamsForRAZOR", None)


def to_torch_name(name):
    """Transform the parameter naming style to PyTorch."""
    if name.startswith("model_"):
        assert name.startswith("model_")
        name = name[6:]
        name = name.replace("_", ".")
    return name


def to_mnm_name(name):
    """Transform the parameter naming style to Meta."""
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
    """A wrapper of torch.autograd.Function to run on Meta."""

    # pylint: disable=no-self-use, unused-argument, missing-docstring, arguments-differ

    @staticmethod
    def forward(ctx, func, *args):
        mod = IRModule.from_expr(func)
        mod = AutoDiff([])(InferType()(mod))
        mod = DeadCodeElimination()(mod)
        mod = CanonicalizeParamsForRAZOR()(InferType()(mod))
        inplace_update_map = InplaceUpdateAnalysis(mod)
        func = mod["main"]
        handle = ValueToHandle(mnm._core.value.ClosureValue({}, func))
        func = _TORCHMNMC._mnm_to_tensor(handle)
        result = _TORCHMNMC._mnm_invoke_relay(func, args, dict(inplace_update_map.items()))
        ctx.bwd = result[1]
        return result[0]

    @staticmethod
    def backward(ctx, grad_output):
        ret = _TORCHMNMC._mnm_invoke_relay(ctx.bwd, [grad_output], {})
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
    func: Callable
        A function to run on Meta.
    """

    def func(*args, **kwargs):
        # TODO: cache the result
        # TODO: eliminate shape_dict
        # TODO: use torch.jit.script
        assert len(args) == 1
        assert not kwargs
        shape_dict = {
            "input0": ((list(args[0].shape), str(args[0].dtype).rsplit(".", maxsplit=1)[-1]))
        }
        cloned_module = copy.deepcopy(module)
        model = mnm.frontend.from_pytorch(cloned_module, shape_dict)
        record = model._internal(mnm.array(asnumpy(args[0])))
        # Difference between state_dict() and named_parameters():
        # module.state_dict() sets the requires_grad of all parameters to false
        # module.state_dict() includes in-place updated parameters while model.paramters() does not
        # so here we use them in combination
        # See also:
        # https://discuss.pytorch.org/t/difference-between-state-dict-and-parameters/37531/8
        # https://discuss.pytorch.org/t/batch-norm-parameters-not-included-in-model-parameters/10265
        named_parameters = dict(module.named_parameters())
        params = {
            k: named_parameters[k] if k in named_parameters else v
            for k, v in module.state_dict().items()
        }
        positional_args = get_positional_args(record.mod["main"], *args, **params)
        return RelayFunction.apply(record.mod["main"], *positional_args)

    return func
