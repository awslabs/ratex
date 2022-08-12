# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JIT script."""
# pylint: disable=c-extension-no-member, protected-access, too-many-locals
import copy
import hashlib
import logging

import torch
import raf
import tvm
from raf import distributed as dist
from raf._ffi.pass_ import AutoDiff, DeadCodeElimination, InferType, Substitute

import _RATEXC

from .._lib import raf
from ..value import ValueToHandle
from ..utils.utils import ltc_timed, to_raf_name, to_torch_name
from ..utils.cache import cache as persist_cache

# pylint: disable=invalid-name
logger = logging.getLogger("jit.script")

_APIS = raf._lib._get_apis()
InplaceUpdateAnalysis = _APIS.get("raf.pass_.InplaceUpdateAnalysis", None)
CanonicalizeParamsForRATEX = _APIS.get("raf.pass_.CanonicalizeParamsForRATEX", None)
ConvertBfFp16Constant = _APIS.get("raf.pass_.ConvertBfFp16Constant", None)
# pylint: enable=invalid-name

TORCH_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}


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
    for name in param_names:
        if name in kwargs:
            param = kwargs[name]
        else:
            param = args[i]
            i = i + 1
        assert isinstance(param, (torch.Tensor, torch.nn.parameter.Parameter))
        ret.append(param)
    assert i == len(args)
    return ret


class RelayFunction(torch.autograd.Function):
    """A wrapper of torch.autograd.Function to run on RAF."""

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
    return x.to(device="cpu").detach().numpy()


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
def convert_module_to_raf(module, shape_n_dtype, args):
    """Convert the PyTorch module to RAF and apply necessary transformations.
    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to be converted.
    shape_n_dtype : List[Tuple[int, torch.dtype]]
        The shape and dtype of the input tensor.
    args : List[torch.Tensor]
        The input tensors.

    Returns
    -------
    ret: Tuple[relay.Function, Dict[str, str], Dict[int, int], Dict[str, raf.array]]
        A tuple of converted function, parameter names, inplace update map, and parameter map
    """
    cloned_module = copy.deepcopy(module)

    bf_fp_16_dtype = None
    for (name, para) in cloned_module.named_parameters():
        if para.dtype == torch.bfloat16:
            bf_fp_16_dtype = "bfloat16"
            break
        if para.dtype == torch.float16:
            bf_fp_16_dtype = "float16"
            break
    # When any bf16/fp16 parameter is found, we assume the dtype of this module is bf16/fp32.
    # In this case, we first convert the module to float32 in order to convert it to a RAF module.
    # Afterward, we apply a pass to make the RAF module bf16/fp32.
    if bf_fp_16_dtype is not None:
        cloned_module.to("cpu")
        cloned_module.float()
        # if the input dtype is bf16/fp16, we make up a float32 input for cpu tracing
        if shape_n_dtype[1] == bf_fp_16_dtype:
            shape_n_dtype = (shape_n_dtype[0], "float32")

    model = raf.frontend.from_pytorch(cloned_module, {"input0": shape_n_dtype})
    raf_params = model.state()
    # ensure raf_params are cachable
    raf_params_shape = {k: v.shape for k, v in raf_params.items()}

    # Must use *.clone(), otherwise the tensor will be removed from live tensors graph
    # because asnumpy() calls *.cpu()
    arg = args[0].clone()
    # if arg is bf16/fp16, we convert it to float32 for tracing
    if bf_fp_16_dtype is not None and arg.dtype in (torch.bfloat16, torch.float16):
        arg = arg.float()
    record = model._internal(raf.array(asnumpy(arg)))
    mod = record.mod

    # if it is a bfloat16 model, we first convert all the float parameters back to bfloat16;
    # then resolve the float constants.
    if bf_fp_16_dtype is not None:
        # fisrt create a new function with bf16/fp16 parameters
        # collect the float32-to-bf16/fp16 mapping
        params, param_map = [], {}
        for para in mod["main"].params:
            name, shape = para.name_hint, para.type_annotation.shape
            dtype = para.type_annotation.dtype
            # The assumption is that when one parameter is bf16/fp32,
            # then all the parameter is bf16/fp16
            if dtype == "float32":
                para_bf_fp_16 = raf._core.ir_ext.extended_var(
                    name, shape=shape, dtype=bf_fp_16_dtype
                )
                params.append(para_bf_fp_16)
                param_map[para] = para_bf_fp_16
            else:
                params.append(para)
        # substitute the float32 parameters with bf16/fp16 parameters
        body_bf_fp_16 = Substitute(mod["main"].body, param_map)
        f_bf_fp_16 = tvm.relay.Function(params=params, body=body_bf_fp_16)
        mod.update_func(mod.get_global_var("main"), f_bf_fp_16)

        # convert float32 constants to bfloat16 constants
        mod = ConvertBfFp16Constant(bf_fp_16_dtype)(mod)

        # if it is a bfloat16 model, we need to modify raf_params_dtype back to bfloat16.
        raf_params_dtype = {
            k: "bfloat16" if v.dtype == "float32" else v.dtype for k, v in raf_params.items()
        }
    else:
        raf_params_dtype = {k: v.dtype for k, v in raf_params.items()}

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
        A function to run on RAF.
    """
    cloned_module = copy.deepcopy(module)
    cloned_module = cloned_module.cpu()

    class ScriptModule(torch.nn.Module):
        """The script module to leverage RAF for compilation and execution.
        Note that all parameters in the original PyTorch module will be flatten in this module,
        as RAF does not need the hierarchical structure of parameters. For example, the original
        model.conv1.weight will become model_conv1_weight. This is intentional because we cannot
        keep the original module as the sub-module of this one for unknown reasons, which result
        in parameters not being updated during training.

        Accordingly, we now use a workaround by introducing an API "native_cpu" that copies
        the trained parameter values to the vanilla PyTorch module on CPU.
        """

        def __init__(self):
            super().__init__()
            self._param_names = []
            for key, value in cloned_module.named_parameters():
                name = to_raf_name(key)
                self._param_names.append(name)
                self.register_parameter(name, value)
            for key, value in cloned_module.named_buffers():
                name = to_raf_name(key)
                self._param_names.append(name)
                self.register_buffer(name, value)

        @ltc_timed("RAFTrace")
        def forward(self, *args, **kwargs):
            """forward computation"""
            # TODO: use torch.jit.script
            assert len(args) == 1, f"Only support single input for now, but got {len(args)}"
            assert not kwargs, "Do not support kwargs yet"
            shape_n_dtype = (list(args[0].shape), str(args[0].dtype).rsplit(".", maxsplit=1)[-1])
            cache_key = (hash_torch_module(cloned_module), str(shape_n_dtype))
            if cache_key in JIT_CACHE:
                # Cache hit.
                func, param_names, inplace_update_map = JIT_CACHE[cache_key]
            else:
                # Cache miss. Generate a RAF function and apply a series of transformations.
                (
                    func,
                    param_names,
                    inplace_update_map,
                    raf_params_shape,
                    raf_params_dtype,
                ) = convert_module_to_raf(cloned_module, shape_n_dtype, args)
                # Convert missing args
                for name in param_names:
                    if name == "input0":
                        continue
                    if name not in self._param_names:
                        self._param_names.append(name)
                        self.register_buffer(
                            name,
                            torch.zeros(
                                raf_params_shape[name],
                                dtype=TORCH_DTYPES.get(raf_params_dtype[name], "float32"),
                            ).to(device=args[0].device),
                        )
                        logger.warning(
                            "%s parameter has been converted from raf.array to torch.Tensor.",
                            name,
                        )
                # Updated cached function, param_names, and inplace update map
                JIT_CACHE[cache_key] = (func, param_names, inplace_update_map)
            lazy_params = {k: getattr(self, k) for k in self._param_names}
            positional_args = get_positional_args(param_names, *args, **lazy_params)
            return RelayFunction.apply(func, inplace_update_map, *positional_args)

        def native_cpu(self):
            """Copy the current parameters to the native module on CPU for inference."""
            cloned_module_state_dict = cloned_module.state_dict()
            for raf_name in self._param_names:
                torch_name = to_torch_name(raf_name)
                if torch_name not in cloned_module_state_dict:
                    continue
                cloned_module_state_dict[torch_name] = getattr(self, raf_name).cpu()
            cloned_module.load_state_dict(cloned_module_state_dict)
            return cloned_module

    return ScriptModule()
