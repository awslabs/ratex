# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ratex._lib import raf
import tvm
from tvm import relay
from raf.ir import ScopeBuilder

_APIS = raf._lib._get_apis()
InplaceUpdateByAlias = _APIS.get("raf.pass_.InplaceUpdateByAlias", None)


def test_basic():
    relu_op = raf._ffi.op.GetOp("raf.op.relu")
    add_op = raf._ffi.op.GetOp("raf.op.add")
    null = raf.ir.const(None)

    def before():
        data_1 = raf.ir.var("p1", shape=(16, 16))
        data_2 = raf.ir.var("p2", shape=(16, 16))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data_1]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, data_2, null, null]))
        out = sb.let("a3", relay.Tuple([a_1, a_2]))
        sb.ret(out)
        func = relay.Function([data_1, data_2], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        data_1 = raf.ir.var("p1", shape=(16, 16))
        data_2 = raf.ir.var("p2", shape=(16, 16))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data_1]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, data_2, data_1, null]))
        out = sb.let("a3", relay.Tuple([a_1, a_2]))
        sb.ret(out)
        func = relay.Function([data_1, data_2], sb.get())
        return tvm.IRModule.from_expr(func)

    mod = before()

    # No alias map. Do nothing.
    mod = InplaceUpdateByAlias({})(mod)
    assert tvm.ir.structural_equal(InplaceUpdateByAlias({})(mod)["main"], mod["main"])

    # Map output.0 to input.1.
    mod = InplaceUpdateByAlias({tvm.tir.IntImm("int32", 0): 1})(mod)
    assert tvm.ir.structural_equal(mod["main"], expected()["main"])


def test_single_out():
    relu_op = raf._ffi.op.GetOp("raf.op.relu")
    add_op = raf._ffi.op.GetOp("raf.op.add")
    null = raf.ir.const(None)

    def before():
        data_1 = raf.ir.var("p1", shape=(16, 16))
        data_2 = raf.ir.var("p2", shape=(16, 16))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data_1]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, data_2, null, null]))
        sb.ret(a_2)
        func = relay.Function([data_1, data_2], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        data_1 = raf.ir.var("p1", shape=(16, 16))
        data_2 = raf.ir.var("p2", shape=(16, 16))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data_1]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, data_2, data_1, null]))
        sb.ret(a_2)
        func = relay.Function([data_1, data_2], sb.get())
        return tvm.IRModule.from_expr(func)

    mod = before()

    # Map output to input.0.
    mod = InplaceUpdateByAlias({tvm.tir.IntImm("int32", 0): 0})(mod)
    assert tvm.ir.structural_equal(mod["main"], expected()["main"])


def test_not_inplace_op():
    relu_op = raf._ffi.op.GetOp("raf.op.relu")
    mul_op = raf._ffi.op.GetOp("raf.op.multiply")

    def before():
        data_1 = raf.ir.var("p1", shape=(16, 16))
        data_2 = raf.ir.var("p2", shape=(16, 16))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data_1]))
        a_2 = sb.let("a2", relay.Call(mul_op, [a_1, data_2]))
        out = sb.let("a3", relay.Tuple([a_1, a_2]))
        sb.ret(out)
        func = relay.Function([data_1, data_2], sb.get())
        return tvm.IRModule.from_expr(func)

    # Attempt to map output.0 to input.1 but failed because multiply cannot be inplace updated.
    mod = InplaceUpdateByAlias({tvm.tir.IntImm("int32", 0): 1})(before())
    assert tvm.ir.structural_equal(mod["main"], before()["main"])


if __name__ == "__main__":
    pytest.main([__file__])
