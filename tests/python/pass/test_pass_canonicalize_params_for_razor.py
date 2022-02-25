# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from razor._lib import mnm
import tvm
from tvm import relay
from mnm.ir import ScopeBuilder
from mnm.model.nn import BatchNorm
from mnm.testing import randn

_APIS = mnm._lib._get_apis()
CanonicalizeParamsForRAZOR = _APIS.get("mnm.pass_.CanonicalizeParamsForRAZOR", None)


def test_basic():
    class Model(mnm.Model):
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @mnm.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            return x

    shape = (2, 3, 4, 5)
    model = Model(num_features=shape[1])
    model.train_mode()
    m_x, _ = randn(shape, requires_grad=True)

    record = model._internal(m_x)
    mod = record.mod
    mod = mnm._ffi.pass_.InferType()(mod)
    mod = mnm._ffi.pass_.AutoDiff(record.requires_grads)(mod)
    mod = mnm._ffi.pass_.DeadCodeElimination()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)
    mod = CanonicalizeParamsForRAZOR()(mod)
    mod = mnm._ffi.pass_.DeadCodeElimination()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)

    def expected():
        bn_op = mnm._ffi.op.GetOp("mnm.op.batch_norm_train")
        bn_dx_op = mnm._ffi.op.GetOp("mnm.op.batch_norm_train_dxwb")
        zeros_like_op = mnm._ffi.op.GetOp("mnm.op.zeros_like")
        eps_const = mnm.ir.const(1e-5)
        momentum_const = mnm.ir.const(0.1)

        data = mnm.ir.var("x", shape=shape)
        bn_b = mnm.ir.var("bn_b", shape=(shape[1],))
        bn_m = mnm.ir.var("bn_m", shape=(shape[1],))
        bn_v = mnm.ir.var("bn_v", shape=(shape[1],))
        bn_w = mnm.ir.var("bn_w", shape=(shape[1],))
        dy = mnm.ir.var("dy", shape=shape)

        # adjoint closure.
        sb = ScopeBuilder()
        x_0 = sb.let("x0", dy)
        x_1 = sb.let("x1", relay.Call(bn_dx_op, [x_0, data, bn_w, bn_b, eps_const]))
        x_2 = sb.let("x2", relay.TupleGetItem(x_1, 0))
        x_3 = sb.let("x3", relay.TupleGetItem(x_1, 1))
        x_4 = sb.let("x4", relay.TupleGetItem(x_1, 2))
        x_5 = sb.let("x5", relay.Call(zeros_like_op, [bn_m]))
        x_6 = sb.let("x5", relay.Call(zeros_like_op, [bn_v]))
        x_7 = sb.let("x7", relay.Tuple([x_2, x_4, x_5, x_6, x_3]))
        sb.ret(x_7)
        closure = relay.Function([dy], sb.get())

        # main function.
        sb = ScopeBuilder()
        a_1 = sb.let(
            "a1",
            relay.Call(bn_op, [data, bn_m, bn_v, bn_w, bn_b, momentum_const, eps_const]),
        )
        a_2 = sb.let("a2", relay.TupleGetItem(a_1, 0))
        a_3 = sb.let("a3", relay.TupleGetItem(a_1, 1))
        a_4 = sb.let("a4", relay.TupleGetItem(a_1, 2))
        adjoint_closure = sb.let("adjoint_closure", closure)
        ret = sb.let("ret", relay.Tuple([a_2, a_3, a_4, adjoint_closure]))
        sb.ret(ret)
        func = relay.Function([data, bn_b, bn_m, bn_v, bn_w], sb.get())
        mod = tvm.IRModule.from_expr(func)
        mod = mnm._ffi.pass_.InferType()(mod)
        return mod

    assert tvm.ir.structural_equal(mod["main"], expected()["main"])


if __name__ == "__main__":
    pytest.main([__file__])
