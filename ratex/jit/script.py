# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JIT script."""
# pylint: disable=c-extension-no-member, protected-access, too-many-locals
import copy
import hashlib
import logging

import _RATEXC
import torch
import raf
import tvm
from raf import distributed as dist
from raf._ffi.pass_ import AutoDiff, DeadCodeElimination, InferType

from .._lib import raf
from ..value import ValueToHandle
from ..utils.utils import ltc_timed
from ..utils.cache import cache as persist_cache

# pylint: disable=invalid-name
logger = logging.getLogger("jit.script")

_APIS = raf._lib._get_apis()
InplaceUpdateAnalysis = _APIS.get("raf.pass_.InplaceUpdateAnalysis", None)
CanonicalizeParamsForRATEX = _APIS.get("raf.pass_.CanonicalizeParamsForRATEX", None)
# pylint: enable=invalid-name

TORCH_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}


def to_torch_name(name):
    """Transform the parameter naming style to PyTorch."""
    if name.startswith("model_"):
        assert name.startswith("model_")
        name = name[len("model_") :]
        name = name.replace("_", ".")
    return name


def to_raf_name(name):
    """Transform the parameter naming style to Meta."""
    return "model_" + name.replace(".", "_")


def get_positional_args(param_names, *args, **kwargs):
    """convert a mixture of positional args and keyword args to positional args only

    Parameters
    ----------
    param_names : List[str]
        A list of parameter names.

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
    mnm_kwargs = {to_raf_name(k): v for k, v in kwargs.items()}
    for name in param_names:
        if name in mnm_kwargs:
            param = mnm_kwargs[name]
        else:
            param = args[i]
            i = i + 1
        assert isinstance(param, (torch.Tensor, torch.nn.parameter.Parameter))
        ret.append(param)
    assert i == len(args)
    return ret


class RelayFunction(torch.autograd.Function):
    """A wrapper of torch.autograd.Function to run on Meta."""

    # pylint: disable=no-self-use, unused-argument, missing-docstring
    # pylint: disable=arguments-differ, abstract-method

    @staticmethod
    def forward(ctx, func, inplace_update_map, *args):
        handle = ValueToHandle(raf._core.value.ClosureValue({}, func))
        t_func = _RATEXC._raf_to_tensor(handle)
        result = _RATEXC._raf_invoke_relay(t_func, args, inplace_update_map)
        ctx.bwd = result[1]
        return result[0]

    @staticmethod
    def backward(ctx, grad_output):
        ret = _RATEXC._raf_invoke_relay(ctx.bwd, [grad_output], {})

        # Each value in the return tuple is a gradient corresponding to the forward input,
        # so the first 2 must be None, because the forward func and inplace_update_map don't
        # have gradients.
        return tuple([None, None] + ret)


def asnumpy(x):
    """Helper function to convert x to numpy"""
    assert isinstance(x, torch.Tensor), f"{type(x)} is not supported"
    return x.cpu().detach().numpy()


def hash_torch_module(module):
    """Hash a PyTorch module to be MD5.

    Parameters
    ----------
    module: torch.nn.Module
        The module to be hashed.

    Returns
    -------
    md5: str
        The MD5 hash of the module.
    """
    return hashlib.md5(str(module).encode(encoding="UTF-8")).hexdigest()


def persist_cache_fn(wrapped_func):
    """Persistent cache a Python function. Note that we assume the cached function
    refers to the distributed context, so it values are also a part of the cache key.

    Parameters
    ----------
    wrapped_func: Callable
        The function to be cached.

    Returns
    -------
    fun: Callable
        The wrapped function with caching.
    """

    def wrapper(module, shape_n_dtype, args):
        comm = dist.get_communicator()
        # params included in cache key to distinguish full-precision and half-precision models
        params = sorted(
            [(name, (param.shape, param.dtype)) for name, param in module.named_parameters()]
        )
        cache_key = (
            hash_torch_module(module),
            str(shape_n_dtype),
            str(tuple(params)),
            comm.size,
            "convert_module_to_raf",
        )

        def unpack(value):
            if isinstance(value, tvm.ir.Array):
                return [unpack(x) for x in value]
            if isinstance(value, tvm.ir.Map):
                return {unpack(k): unpack(v) for k, v in value.items()}
            return value

        def loader(value):
            func, param_names, inplace_update_map, raf_params_shape, raf_params_dtype = unpack(
                raf.ir.load_json(value)
            )
            # raf_params_shape is tuple instead of list
            raf_params_shape = {k: tuple([e.value for e in v]) for k, v in raf_params_shape.items()}
            return func, param_names, inplace_update_map, raf_params_shape, raf_params_dtype

        def saver(value):
            return raf.ir.save_json(value)

        value = persist_cache.query(cache_key, loader=loader)
        if value is None:
            value = wrapped_func(module, shape_n_dtype, args)
            persist_cache.commit(cache_key, value, saver=saver)
        return value

    return wrapper


@ltc_timed("RAFTraceConvertModuleToRAF")
@persist_cache_fn
def convert_module_to_raf(module, shape_n_dtype, arg_np):
    """Convert the PyTorch module to RAF and apply necessary transformations.
    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to be converted.
    shape_n_dtype : List[Tuple[int, torch.dtype]]
        The shape and dtype of the input tensor.
    arg_np : np.ndarray
        The input tensors in numpy array. Note that we do not support multiple arguments for now.

    Returns
    -------
    ret: Tuple[relay.Function, Dict[str, str], Dict[int, int], Dict[str, raf.array]]
        A tuple of converted function, parameter names, inplace update map, and parameter map
    """
    cloned_module = copy.deepcopy(module)

    model = raf.frontend.from_pytorch(cloned_module, {"input0": shape_n_dtype})
    raf_params = model.state()
    # ensure raf_params are cachable
    raf_params_shape = {k: v.shape for k, v in raf_params.items()}
    raf_params_dtype = {k: v.dtype for k, v in raf_params.items()}

    # Must use *.clone(), otherwise the tensor will be removed from live tensors graph
    # because asnumpy() calls *.cpu()
    record = model._internal(raf.array(arg_np))
    mod = record.mod
    mod = AutoDiff([])(InferType()(mod))
    mod = DeadCodeElimination()(mod)
    mod = CanonicalizeParamsForRATEX()(InferType()(mod))
    inplace_update_map = dict(InplaceUpdateAnalysis(mod).items())
    mod = InferType()(mod)
    func = mod["main"]
    param_names = [var.name_hint for var in func.params]
    return func, param_names, inplace_update_map, raf_params_shape, raf_params_dtype


# Module based cache that maps the input shape/dtype to a tuple of
# (processed Relay function, function parameter names, inplace update map).
JIT_CACHE = {}


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

    @ltc_timed("RAFTrace")
    def wrapper(*args, **kwargs):
        # TODO: use torch.jit.script
        assert len(args) == 1, f"Only support single input for now, but got {len(args)}"
        assert not kwargs, "Do not support kwargs yet"
        arg0_np = asnumpy(args[0].clone())
        shape_n_dtype = (list(arg0_np.shape), str(arg0_np.dtype).rsplit(".", maxsplit=1)[-1])
        cache_key = (hash_torch_module(module), str(shape_n_dtype))
        if cache_key in JIT_CACHE:
            # Cache hit.
            func, param_names, inplace_update_map = JIT_CACHE[cache_key]
        else:
            # Cache miss. Generate a Meta function and apply a series of transformations.
            (
                func,
                param_names,
                inplace_update_map,
                raf_params_shape,
                raf_params_dtype,
            ) = convert_module_to_raf(module, shape_n_dtype, arg0_np)
            # Convert missing args
            params_keys = [to_raf_name(k) for k in params.keys()]
            for name in param_names:
                if name == "input0":
                    continue
                if name not in params_keys:
                    t_name = to_torch_name(name)
                    params[t_name] = torch.zeros(
                        raf_params_shape[name],
                        dtype=TORCH_DTYPES.get(raf_params_dtype[name], "float32"),
                    ).to("lazy")
                    logger.warning(
                        "%s parameter has been converted from raf.array to torch.Tensor.", name
                    )
            # Updated cached function, param_names, and inplace update map
            JIT_CACHE[cache_key] = (func, param_names, inplace_update_map)

        positional_args = get_positional_args(param_names, *args, **params)
        return RelayFunction.apply(func, inplace_update_map, *positional_args)

    return wrapper
