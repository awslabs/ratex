# import torch_mnm
# from torch_mnm._lib import tvm
# from torch_mnm._lib import mnm

import os
import pathlib

import numpy as np
import torch

import tvm
from tvm import topi
import mnm

import neuroncc
from neuroncc.starfish.penguin.frontends import RelayFE
from neuroncc.starfish.penguin.frontends import MetaFE
from neuroncc.starfish.penguin.frontends import ComputeDefs
from neuroncc.starfish.penguin.ir import ir
from neuroncc.starfish.support import LogContext

const_cnt = 0

# for the case where mnm.add(tensor(), tensor())
# the tensor() is a constant
# but it should not be lowered to ScalarValue
# instead, should be lowered to tensor
# because when add is invoked, tensor().shape is accessed
# ScalarValue does not have a shape
def customized_visit_constant(self, rconst):
    is_tensor_value = False
    try:
        import mnm
        value = mnm._ffi.ir.constant.ExtractValue(rconst).asnumpy()
        is_tensor_value = True
    except:
        value = rconst.data.asnumpy()

    dtype = value.dtype

    if not value.shape and not is_tensor_value:
        return ir.ScalarValue(value=value.tolist(), dtype=dtype)

    if not self.fuse_param_to_neff:
        # constants do not have name_hint
        global const_cnt
        name = "const" + str(const_cnt)
        const_cnt = const_cnt + 1
        shape = value.shape

        # save to npy file
        np.save("value_{}".format(name.replace("/", "__")), value)
        t = self.builder.tensor(name=name, dtype=dtype,
                                shape=shape)
        self.cu.markInput(t)
        self.module.markInput(t)
        return t

    return self.builder.tensor(dtype=dtype, shape=value.shape, value=value)


def print_debug(*args, **kwargs):
  if LogContext.LogContext.instance is not None:
    LogContext.LogContext.instance.debug(*args, **kwargs)
  else:
    print(*args, **kwargs)


def print_error(*args, **kwargs):
  if LogContext.LogContext.instance is not None:
    LogContext.LogContext.instance.error(*args, **kwargs)
  else:
    print(*args, **kwargs)


RelayFE.RelayFE.visit_constant = customized_visit_constant
MetaFE.MetaFE.visit_constant = customized_visit_constant
LogContext.print_error = print_error
LogContext.print_debug = print_debug
ComputeDefs.TVMCompute(name="mnm.op.repeat", operands=['data'],
                       repeats=ComputeDefs.int_attr('repeats'),
                       axis=ComputeDefs.int_attr('axis'))

def randn(shape, *, dtype="float32", std=1.0):
    x = np.random.randn(*shape) * std
    # x = np.zeros(shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x)
    t_x = torch.from_numpy(x)
    return m_x, t_x


def make_value(ty):
    return randn(topi.utils.get_const_tuple(ty.shape), dtype=str(ty.dtype))[0]


def post_process_meta_record(func):
    from mnm._ffi import ir
    from mnm._ffi.pass_ import InferType, AutoDiff
    from mnm._ffi.pass_.contrib import AnnotateName
    import tvm

    def _infer_type(func):
        main = tvm.relay.GlobalVar("main")
        mod = tvm.IRModule({main: func})
        mod = InferType()(mod)
        return mod["main"]

    func = AnnotateName(func)
    return _infer_type(func)


# sunda_path = pathlib.Path(__file__).parent.resolve()
# module_path = os.path.join(sunda_path, "module.json")
# model_states_index_path = os.path.join(sunda_path, "module.json")

dirpath = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dirpath, "module.json")) as module_file:
    module_json = module_file.read()
    module = tvm.ir.load_json(module_json)
with open(os.path.join(dirpath, "model_states_index.json")) as model_states_index_file:
    model_states_index_json = model_states_index_file.read()
    model_states_index = tvm.ir.load_json(model_states_index_json)
    model_states_index = topi.utils.get_const_tuple(model_states_index)

# func is tvm.relay.function.Function
func = module["main"]
func = post_process_meta_record(func)
# print("=" * 80)
# print("func:")
# print(type(func))
# print(func.params)
# print(mnm._ffi.ir.AsText(func))

# params is dict: Str -> ndarray
# parameter name (like model_model_conv1_weight) -> ndarray
params = {
    func.params[idx].name_hint: make_value(func.params[idx].checked_type)
    for idx in model_states_index
}
# print("=" * 80)
# print("params:")
# print(type(params))
# print(params.keys())

# reference_inputs is list
# [(Str, ndarray)], like [('input0', ndarray), ('ytrue', ndarray)]
reference_inputs = [
    (param.name_hint, make_value(param.checked_type))
    for idx, param in enumerate(func.params) if idx not in model_states_index
]
# print("=" * 80)
# print("reference_inputs:")
# print(type(reference_inputs))
# print([k for k, v in reference_inputs])


# inference_output_names = [
#     'x_81', 'x_85', 'x_92', 'x_96', 'x_101',
#     'x_105', 'x_110', 'x_114', 'x_119', 'x_123', 'p12', 'x_17',
#     'x_124', 'x_72', 'x_73', 'x_71', 'x_67', 'x_65', 'x_60',
#     'x_58', 'x_55', 'x_53', 'x_50', 'x_48', 'x_73', 'x_71', 'x_67',
#     'x_65', 'x_60', 'x_58', 'x_55', 'x_53', 'x_50', 'x_48'
# ]
# gradient_output_names is list
# [(Str, ndarray)], like [('input0.grad', ndarray), ('model_model_conv1_weight.grad', ndarray), ...]
# Note: 
# 1. all ndarray (both of data inputs and param inputs) which requires grad are included here
# 2. the ndarray value is the same as that of `params`
# print("=" * 80)
# print("gradient_output_names:")
# print(type(gradient_output_names))
# print(gradient_output_names)
