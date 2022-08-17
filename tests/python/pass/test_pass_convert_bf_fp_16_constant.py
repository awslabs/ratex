# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ratex._lib import raf
from raf.ir import ScopeBuilder
import tvm
from tvm import relay
import numpy as np

_APIS = raf._lib._get_apis()
ConvertBfFp16Constant = _APIS.get("raf.pass_.ConvertBfFp16Constant", None)

# the pass should cast all the constant float32 tensors to constant bf16/fp16 tensors;
@pytest.mark.parametrize("bf_fp_16_dtype", ["bfloat16", "float16"])
def test_basic_cast_constant(bf_fp_16_dtype):
    def before():
        mul_op = raf._ffi.op.GetOp("raf.op.multiply")
        data = raf.ir.var("input", shape=(1, 128), dtype="bfloat16")
        const = raf.ir.const(np.array(168).astype(np.float32))
        sb = ScopeBuilder()
        a_0 = sb.let("a0", const)
        a_1 = sb.let("a1", relay.Call(mul_op, [data, a_0]))
        sb.ret(a_1)
        func = relay.Function([data], sb.get())
        mod = tvm.IRModule.from_expr(func)
        return ConvertBfFp16Constant(bf_fp_16_dtype)(mod)

    def expected():
        mul_op = raf._ffi.op.GetOp("raf.op.multiply")
        cast_op = raf._ffi.op.GetOp("raf.op.cast")
        data = raf.ir.var("input", shape=(1, 128), dtype="bfloat16")
        const = raf.ir.const(np.array(168).astype(np.float32))
        cast_type = raf.ir.const(bf_fp_16_dtype)
        sb = ScopeBuilder()
        a_0 = sb.let("a0", const)
        a_0_cast = sb.let("a0_" + bf_fp_16_dtype, relay.Call(cast_op, [a_0, cast_type]))
        a_1 = sb.let("a1", relay.Call(mul_op, [data, a_0_cast]))
        sb.ret(a_1)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    assert tvm.ir.structural_equal(before()["main"], expected()["main"])


# if a cast call's target dtype is float32, the pass should change the target dtype to bf16/fp16
@pytest.mark.parametrize("bf_fp_16_dtype", ["bfloat16", "float16"])
def test_basic_cast_target(bf_fp_16_dtype):
    def before():
        cast_op = raf._ffi.op.GetOp("raf.op.cast")
        data = raf.ir.var("input", shape=(1, 128), dtype="int")
        cast_type = raf.ir.const("float32")
        sb = ScopeBuilder()
        a_0 = sb.let("a0", relay.Call(cast_op, [data, cast_type]))
        sb.ret(a_0)
        func = relay.Function([data], sb.get())
        mod = tvm.IRModule.from_expr(func)
        return ConvertBfFp16Constant(bf_fp_16_dtype)(mod)

    def expected():
        cast_op = raf._ffi.op.GetOp("raf.op.cast")
        data = raf.ir.var("input", shape=(1, 128), dtype="int")
        cast_type = raf.ir.const(bf_fp_16_dtype)
        sb = ScopeBuilder()
        a_0 = sb.let("a0", relay.Call(cast_op, [data, cast_type]))
        sb.ret(a_0)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    assert raf.ir.AsText(before()) == raf.ir.AsText(expected())


if __name__ == "__main__":
    pytest.main([__file__])
