# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ratex._lib import raf
import tvm
from tvm import relay
from raf.ir import ScopeBuilder

_APIS = raf._lib._get_apis()
InplaceUpdateAnalysis = _APIS.get("raf.pass_.InplaceUpdateAnalysis", None)


def test_basic():
    relu_op = raf._ffi.op.GetOp("raf.op.relu")
    add_op = raf._ffi.op.GetOp("raf.op.add")
    null = raf.ir.const(None)

    def get_mod():
        data_1 = raf.ir.var("p1", shape=(16, 16))
        data_2 = raf.ir.var("p2", shape=(16, 16))

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
