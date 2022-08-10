# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from ratex._lib import raf
from raf.ir import ScopeBuilder
from raf.testing import check, get_vm_executor, numpy
from ratex.optimizer import Adam
from ratex.testing import train, fake_image_dataset, verify
from ratex.testing.common import with_seed

import tvm
from tvm import relay

import pytest


@pytest.mark.xfail(reason="raf does not compute bf16/fp16 correctly")
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@with_seed(0)
def test_linear_bf_fp_16_from_raf(dtype):
    def np_float2np_bf16(arr):
        """Convert a numpy array of float to a numpy array
        of bf16 in uint16"""
        orig = arr.view("<u4")
        bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
        return np.right_shift(orig + bias, 16).astype("uint16")

    def np_bf162np_float(arr):
        """Convert a numpy array of bf16 (uint16) to a numpy array
        of float"""
        u32 = np.left_shift(arr.astype("uint32"), 16)
        return u32.view("<f4")

    def res_from_raf(x, w, b):
        matmul_op = raf._ffi.op.GetOp("raf.op.matmul_nt")
        add_op = raf._ffi.op.GetOp("raf.op.bias_add")

        data_0 = raf.ir.var("input", shape=(1, 128), dtype=dtype)
        data_1 = raf.ir.var("w", shape=(128, 128), dtype=dtype)
        data_2 = raf.ir.var("b", shape=(128,), dtype=dtype)

        sb = ScopeBuilder()
        a_2 = sb.let("a2", relay.Call(matmul_op, [data_0, data_1]))
        a_3 = sb.let("a3", relay.Call(add_op, [a_2, data_2]))
        sb.ret(a_3)
        func = relay.Function([data_0, data_1, data_2], sb.get())
        mod = tvm.IRModule.from_expr(func)

        vm = get_vm_executor(mod, "cpu")
        return vm(x, w, b)

    def res_from_py(x, w, b):
        mod = torch.nn.Linear(128, 128)
        with torch.no_grad():
            mod.weight = torch.nn.Parameter(w)
            mod.bias = torch.nn.Parameter(b)
        return mod(x)

    x = np.random.randn(1, 128).astype("float32")
    w = np.random.randn(128, 128).astype("float32")
    b = np.random.randn(128).astype("float32")

    if dtype == "bfloat16":
        x_raf = raf.array(np_float2np_bf16(x), device="cpu")
        w_raf = raf.array(np_float2np_bf16(w), device="cpu")
        b_raf = raf.array(np_float2np_bf16(b), device="cpu")
    else:
        assert dtype == "float16", "unexpected dtype"
        x_raf = raf.array(x.astype(np.float16), device="cpu")
        w_raf = raf.array(w.astype(np.float16), device="cpu")
        b_raf = raf.array(b.astype(np.float16), device="cpu")

    out_raf = res_from_raf(x_raf, w_raf, b_raf)

    if dtype == "bfloat16":
        out_raf = np_bf162np_float(numpy(out_raf))

    dtype_pt = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    x_pt = torch.from_numpy(x).to("cuda").type(dtype_pt)
    w_pt = torch.from_numpy(w).to("cuda").type(dtype_pt)
    b_pt = torch.from_numpy(b).to("cuda").type(dtype_pt)
    out_pt = res_from_py(x_pt, w_pt, b_pt)
    out_pt = numpy(out_pt.float())

    check(out_raf, out_pt, rtol=1e-3, atol=1e-3)


# Currently, bf16 may fail randomly but fp16 should pass.
@pytest.mark.skip(reason="raf does not compute bf16/fp16 correctly")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@with_seed(0)
def test_linear_bf16_from_pt(dtype):
    class SingleLayerLogistics(torch.nn.Module):
        def __init__(self, input_shape=28, num_classes=10):
            super(SingleLayerLogistics, self).__init__()
            self.log_softmax = torch.nn.LogSoftmax(dim=-1)
            self.linear = torch.nn.Linear(784, num_classes)

        def forward(self, x):
            out = torch.flatten(x, 1)
            out = self.linear(out)
            out = torch.relu(out)
            out = self.log_softmax(out)
            return out

    num_classes = 10
    batch_size = 1

    dataset = fake_image_dataset(batch_size, 1, 28, 10, dtype=dtype)
    model = SingleLayerLogistics(num_classes=num_classes).to(dtype=dtype)
    lazy_results = train(
        "lazy",
        model,
        dataset,
        dtype=dtype,
        optimizer=Adam,
        num_classes=num_classes,
        batch_size=batch_size,
    )
    cpu_results = train(
        "cuda",
        model,
        dataset,
        dtype=dtype,
        optimizer=Adam,
        num_classes=num_classes,
        batch_size=batch_size,
    )
    verify(lazy_results, cpu_results, tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
