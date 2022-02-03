"""The JIT script."""
# pylint: disable=c-extension-no-member, protected-access, too-many-locals
import copy
import hashlib
import logging

import torch
import mnm
import tvm
from mnm import distributed as dist
from mnm._ffi.pass_ import AutoDiff, AutoDataParallel, DeadCodeElimination, InferType

from .. import _TORCHMNMC
from .._lib import mnm
from ..value import ValueToHandle
from ..utils.utils import ltc_timed
from ..utils.cache import cache as persist_cache

logger = logging.getLogger("jit.script")

_APIS = mnm._lib._get_apis()
InplaceUpdateAnalysis = _APIS.get("mnm.pass_.InplaceUpdateAnalysis", None)
CanonicalizeParamsForRAZOR = _APIS.get("mnm.pass_.CanonicalizeParamsForRAZOR", None)

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


def to_mnm_name(name):
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
    mnm_kwargs = {to_mnm_name(k): v for k, v in kwargs.items()}
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

    # pylint: disable=no-self-use, unused-argument, missing-docstring, arguments-differ

    @staticmethod
    def forward(ctx, func, inplace_update_map, *args):
        handle = ValueToHandle(mnm._core.value.ClosureValue({}, func))
        t_func = _TORCHMNMC._mnm_to_tensor(handle)
        result = _TORCHMNMC._mnm_invoke_relay(t_func, args, inplace_update_map)
        ctx.bwd = result[1]
        return result[0]

    @staticmethod
    def backward(ctx, grad_output):
        ret = _TORCHMNMC._mnm_invoke_relay(ctx.bwd, [grad_output], {})

        # Each value in the return tuple is a gradient corresponding to the forward input,
        # so the first 2 must be None, because the forward func and inplace_update_map don't
        # have gradients.
        return tuple([None, None] + ret)


def asnumpy(x):
    """Helper function to convert x to numpy"""
    assert isinstance(x, torch.Tensor), f"{type(x)} is not supported"
    return x.cpu().detach().numpy()


def persis_cache_fn(wrapped_func):
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
        dctx = dist.get_context()
        cache_key = (
            hashlib.md5(str(module).encode(encoding="UTF-8")).hexdigest(),
            str(shape_n_dtype),
            dctx.enable_data_parallel,
            dctx.size,
            "convert_module_to_meta",
        )

        def unpack(value):
            if isinstance(value, tvm.ir.Array):
                return [unpack(x) for x in value]
            if isinstance(value, tvm.ir.Map):
                return {unpack(k): unpack(v) for k, v in value.items()}
            return value

        def loader(value):
            func, param_names, inplace_update_map, mnm_params_shape, mnm_params_dtype = \
                unpack(mnm.ir.load_json(value))
            # mnm_params_shape is tuple instead of list
            mnm_params_shape = {k: tuple(v) for k, v in mnm_params_shape.items()}
            return func, param_names, inplace_update_map, mnm_params_shape, mnm_params_dtype

        def saver(value):
            return mnm.ir.save_json(value)

        value = persist_cache.query(cache_key, loader=loader)
        if value is None:
            value = wrapped_func(module, shape_n_dtype, args)
            persist_cache.commit(cache_key, value, saver=saver)
        return value
    return wrapper


@ltc_timed("MNMTraceConvertModuleToMeta")
@persis_cache_fn
def convert_module_to_meta(module, shape_n_dtype, args):
    """Convert the PyTorch module to Meta and apply necessary transformations.
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
    ret: Tuple[relay.Function, Dict[str, str], Dict[int, int], Dict[str, mnm.array]]
        A tuple of converted function, parameter names, inplace update map, and parameter map
    """
    dctx = dist.get_context()
    cloned_module = copy.deepcopy(module)
    cache_key = (
        hashlib.md5(str(cloned_module).encode(encoding="UTF-8")).hexdigest(),
        str(shape_n_dtype),
    )
    cached_model_path = persist_cache.query(cache_key)

    # Cache miss. Add new entries to the cache and directly let from_pytorch write the
    # traced model to the persistent storage.
    if cached_model_path is None:
        cached_model_path = persist_cache.create_entry(cache_key)

    cached_model_file, cached_hash_file = None, None
    if cached_model_path is not None:
        cached_model_file = cached_model_path / "model.pt"
        cached_hash_file = cached_model_path / "model.hash"

    model = mnm.frontend.from_pytorch(
        cloned_module,
        {"input0": shape_n_dtype},
        model_file=cached_model_file,
        hash_file=cached_hash_file,
    )
    mnm_params = model.state()
    # ensure mnm_params are cachable
    mnm_params_shape = {
        k: v.shape for k, v in mnm_params.items()
    }
    mnm_params_dtype = {
        k: v.dtype for k, v in mnm_params.items()
    }

    record = model._internal(mnm.array(asnumpy(args[0])))
    mod = record.mod
    mod = AutoDiff([])(InferType()(mod))
    if dctx.enable_data_parallel:
        # FIXME: move AutoDataParallel to Optimize in client files
        mod = AutoDataParallel()(mod)
    mod = DeadCodeElimination()(mod)
    mod = CanonicalizeParamsForRAZOR()(InferType()(mod))
    inplace_update_map = dict(InplaceUpdateAnalysis(mod).items())
    func = mod["main"]
    param_names = [var.name_hint for var in func.params]
    return func, param_names, inplace_update_map, mnm_params_shape, mnm_params_dtype


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
    # Module based cache that maps the input shape/dtype to a tuple of
    # (processed Relay function, function parameter names, inplace update map).
    cache = {}

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

    @ltc_timed("MNMTrace")
    def func(*args, **kwargs):
        # TODO: eliminate shape_dict
        # TODO: use torch.jit.script
        assert len(args) == 1, f"Only support single input for now, but got {len(args)}"
        assert not kwargs, "Do not support kwargs yet"
        shape_n_dtype = (list(args[0].shape), str(args[0].dtype).rsplit(".", maxsplit=1)[-1])
        cache_key = str(shape_n_dtype)
        if cache_key in cache:
            # Cache hit.
            func, param_names, inplace_update_map = cache[cache_key]
        else:
            # Cache miss. Generate a Meta function and apply a series of transformations.
            func, param_names, inplace_update_map, mnm_params_shape, mnm_params_dtype = \
                convert_module_to_meta(module, shape_n_dtype, args)
            # Convert missing args
            params_keys = [to_mnm_name(k) for k in params.keys()]
            for name in param_names:
                if name == "input0":
                    continue
                if name not in params_keys:
                    t_name = to_torch_name(name)
                    params[t_name] = torch.zeros(
                        mnm_params_shape[name],
                        dtype=TORCH_DTYPES.get(mnm_params_dtype[name], "float32"),
                    ).to("xla")
                    logger.warning(
                        "%s parameter has been converted from mnm.array to torch.Tensor.", name
                    )
            # Updated cached function, param_names, and inplace update map
            cache[cache_key] = (func, param_names, inplace_update_map)

        positional_args = get_positional_args(param_names, *args, **params)
        return RelayFunction.apply(func, inplace_update_map, *positional_args)

    return func
