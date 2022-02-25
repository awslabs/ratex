# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from razor._lib import mnm
import tvm
from tvm import relay
from mnm.ir import ScopeBuilder

_APIS = mnm._lib._get_apis()
InplaceUpdateAnalysis = _APIS.get("mnm.pass_.InplaceUpdateAnalysis", None)


def test_basic():
    relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
    add_op = mnm._ffi.op.GetOp("mnm.op.add")
    null = mnm.ir.const(None)

    def get_mod():
        data_1 = mnm.ir.var("p1", shape=(16, 16))
        data_2 = mnm.ir.var("p2", shape=(16, 16))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data_1]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, data_2, null, null]), may_share=data_1)
        out = sb.let("a3", relay.Tuple([a_1, a_2]))
        sb.ret(out)
        func = relay.Function([data_1, data_2], sb.get())
        return tvm.IRModule.from_expr(func)

    mod = get_mod()
    ret = dict(InplaceUpdateAnalysis(mod).items())
    expected = {1: 0}
    assert ret == expected


if __name__ == "__main__":
    pytest.main([__file__])
