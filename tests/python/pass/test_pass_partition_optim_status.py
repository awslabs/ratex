# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from razor._lib import raf
from raf.ir import ScopeBuilder
from raf._ffi.pass_ import InferType

import tvm
from tvm import relay

_APIS = raf._lib._get_apis()
PartitionOptimStatus = _APIS.get("raf.pass_.PartitionOptimStatus", None)


def test_basic():
    # Set mock distributed context
    dctx = raf.distributed.get_context()
    dctx.rank = 1
    dctx.size = 2
    dctx.zero_opt_level = 1

    strided_slice_op = raf._ffi.op.GetOp("raf.op.strided_slice")
    scatter_op = raf._ffi.op.GetOp("raf.op.scatter")
    add_op = raf._ffi.op.GetOp("raf.op.add")
    allgather_op = raf._ffi.op.GetOp("raf.op._allgather")

    shape_1 = (16, 16)
    shape_1_sliced = (8, 16)

    shape_2 = (8, 17)
    shape_2_sliced = (4, 17)

    def before():
        w_1 = raf.ir.var("w1", shape=shape_1)
        p_1 = raf.ir.var("p1", shape=shape_1_sliced)
        idx_1 = raf.ir.const(np.zeros(shape_1_sliced, dtype="int64"))

        w_2 = raf.ir.var("w2", shape=shape_2)
        p_2 = raf.ir.var("p2", shape=shape_2_sliced)
        idx_2 = raf.ir.const(np.zeros(shape_2_sliced, dtype="int64"))

        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(strided_slice_op, [w_1, raf.ir.const(8), raf.ir.const(16), raf.ir.const(1)]),
        )
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1]))
        a_3 = sb.let("a3", relay.Call(scatter_op, [w_1, idx_1, a_2, raf.ir.const(0)]))

        a_4 = sb.let(
            "a4",
            relay.Call(strided_slice_op, [w_2, raf.ir.const(4), raf.ir.const(8), raf.ir.const(1)]),
        )
        a_5 = sb.let("a5", relay.Call(add_op, [a_4, p_2]))
        a_6 = sb.let("a6", relay.Call(scatter_op, [w_2, idx_2, a_5, raf.ir.const(0)]))

        out = sb.let("a7", relay.Tuple([a_3, a_6]))
        sb.ret(out)
        func = relay.Function([w_1, p_1, w_2, p_2], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        w_1 = raf.ir.var("w1", shape=shape_1)
        p_1 = raf.ir.var("p1", shape=shape_1_sliced)

        w_2 = raf.ir.var("w2", shape=shape_2)
        p_2 = raf.ir.var("p2", shape=shape_2_sliced)

        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(strided_slice_op, [w_1, raf.ir.const(8), raf.ir.const(16), raf.ir.const(1)]),
        )
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1]))
        x_0 = sb.let("x_0", relay.Call(allgather_op, [a_2, raf.ir.const(0)]))
        a_3 = sb.let("a3", x_0)

        a_4 = sb.let(
            "a4",
            relay.Call(strided_slice_op, [w_2, raf.ir.const(4), raf.ir.const(8), raf.ir.const(1)]),
        )
        a_5 = sb.let("a5", relay.Call(add_op, [a_4, p_2]))
        x_01 = sb.let("x_01", relay.Call(allgather_op, [a_5, raf.ir.const(0)]))
        a_6 = sb.let("a6", x_01)

        out = sb.let("a7", relay.Tuple([a_3, a_6]))
        sb.ret(out)
        func = relay.Function([w_1, p_1, w_2, p_2], sb.get())
        return tvm.IRModule.from_expr(func)

    mod = before()
    mod = InferType()(mod)
    mod = PartitionOptimStatus()(mod)
    mod = InferType()(mod)

    mod_expected = expected()
    mod_expected = InferType()(mod_expected)

    assert tvm.ir.structural_equal(mod["main"], mod_expected["main"])


def test_uneven_partition():
    # Set mock distributed context
    dctx = raf.distributed.get_context()
    dctx.rank = 1
    dctx.size = 3
    dctx.zero_opt_level = 1

    strided_slice_op = raf._ffi.op.GetOp("raf.op.strided_slice")
    scatter_op = raf._ffi.op.GetOp("raf.op.scatter")
    add_op = raf._ffi.op.GetOp("raf.op.add")
    allgather_op = raf._ffi.op.GetOp("raf.op._allgather")

    shape_1 = (16, 16)
    shape_1_sliced = (6, 16)

    shape_2 = (8, 17)
    shape_2_sliced = (3, 17)

    def before():
        w_1 = raf.ir.var("w1", shape=shape_1)
        p_1 = raf.ir.var("p1", shape=shape_1_sliced)
        idx_1 = raf.ir.const(np.zeros(shape_1_sliced, dtype="int64"))

        w_2 = raf.ir.var("w2", shape=shape_2)
        p_2 = raf.ir.var("p2", shape=shape_2_sliced)
        idx_2 = raf.ir.const(np.zeros(shape_2_sliced, dtype="int64"))

        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(strided_slice_op, [w_1, raf.ir.const(6), raf.ir.const(12), raf.ir.const(1)]),
        )
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1]))
        a_3 = sb.let("a3", relay.Call(scatter_op, [w_1, idx_1, a_2, raf.ir.const(0)]))

        a_4 = sb.let(
            "a4",
            relay.Call(strided_slice_op, [w_2, raf.ir.const(3), raf.ir.const(6), raf.ir.const(1)]),
        )
        a_5 = sb.let("a5", relay.Call(add_op, [a_4, p_2]))
        a_6 = sb.let("a6", relay.Call(scatter_op, [w_2, idx_2, a_5, raf.ir.const(0)]))

        out = sb.let("a7", relay.Tuple([a_3, a_6]))
        sb.ret(out)
        func = relay.Function([w_1, p_1, w_2, p_2], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        w_1 = raf.ir.var("w1", shape=shape_1)
        p_1 = raf.ir.var("p1", shape=shape_1_sliced)

        w_2 = raf.ir.var("w2", shape=shape_2)
        p_2 = raf.ir.var("p2", shape=shape_2_sliced)

        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(strided_slice_op, [w_1, raf.ir.const(6), raf.ir.const(12), raf.ir.const(1)]),
        )
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1]))
        x_0 = sb.let("x_0", relay.Call(allgather_op, [a_2, raf.ir.const(0)]))
        x_1 = sb.let(
            "x_1",
            relay.Call(strided_slice_op, [x_0, raf.ir.const(0), raf.ir.const(16), raf.ir.const(1)]),
        )
        a_3 = sb.let("a3", x_1)

        a_4 = sb.let(
            "a4",
            relay.Call(strided_slice_op, [w_2, raf.ir.const(3), raf.ir.const(6), raf.ir.const(1)]),
        )
        a_5 = sb.let("a5", relay.Call(add_op, [a_4, p_2]))
        x_01 = sb.let("x_01", relay.Call(allgather_op, [a_5, raf.ir.const(0)]))
        x_11 = sb.let(
            "x_11",
            relay.Call(strided_slice_op, [x_01, raf.ir.const(0), raf.ir.const(8), raf.ir.const(1)]),
        )
        a_6 = sb.let("a6", x_11)

        out = sb.let("a7", relay.Tuple([a_3, a_6]))
        sb.ret(out)
        func = relay.Function([w_1, p_1, w_2, p_2], sb.get())
        return tvm.IRModule.from_expr(func)

    mod = before()
    mod = InferType()(mod)
    mod = PartitionOptimStatus()(mod)
    mod = InferType()(mod)

    mod_expected = expected()
    mod_expected = InferType()(mod_expected)

    assert tvm.ir.structural_equal(mod["main"], mod_expected["main"])


def test_single_output():
    # Set mock distributed context
    dctx = raf.distributed.get_context()
    dctx.rank = 1
    dctx.size = 2
    dctx.zero_opt_level = 1

    strided_slice_op = raf._ffi.op.GetOp("raf.op.strided_slice")
    scatter_op = raf._ffi.op.GetOp("raf.op.scatter")
    add_op = raf._ffi.op.GetOp("raf.op.add")
    allgather_op = raf._ffi.op.GetOp("raf.op._allgather")

    shape_1 = (16, 16)
    shape_1_sliced = (8, 16)

    def before():
        w_1 = raf.ir.var("w1", shape=shape_1)
        p_1 = raf.ir.var("p1", shape=shape_1_sliced)
        idx_1 = raf.ir.const(np.zeros(shape_1_sliced, dtype="int64"))

        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(strided_slice_op, [w_1, raf.ir.const(8), raf.ir.const(16), raf.ir.const(1)]),
        )
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1]))
        a_3 = sb.let("a3", relay.Call(scatter_op, [w_1, idx_1, a_2, raf.ir.const(0)]))

        sb.ret(a_3)
        func = relay.Function([w_1, p_1], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        w_1 = raf.ir.var("w1", shape=shape_1)
        p_1 = raf.ir.var("p1", shape=shape_1_sliced)

        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(strided_slice_op, [w_1, raf.ir.const(8), raf.ir.const(16), raf.ir.const(1)]),
        )
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1]))
        x_0 = sb.let("x_0", relay.Call(allgather_op, [a_2, raf.ir.const(0)]))
        a_3 = sb.let("a3", x_0)

        sb.ret(a_3)
        func = relay.Function([w_1, p_1], sb.get())
        return tvm.IRModule.from_expr(func)

    mod = before()
    mod = InferType()(mod)
    mod = PartitionOptimStatus()(mod)
    mod = InferType()(mod)

    mod_expected = expected()
    mod_expected = InferType()(mod_expected)

    assert tvm.ir.structural_equal(mod["main"], mod_expected["main"])


if __name__ == "__main__":
    pytest.main([__file__])
