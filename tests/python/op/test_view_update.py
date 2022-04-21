# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from razor.testing import check


def test_view_data_update():
    a = torch.zeros(4, device="lazy")
    v = a.view(2, 2)
    a.data = a.data + 1
    check(a, np.ones(4, dtype="float32"))
    # Upadting a.data should not update v's value.
    check(v, np.zeros((2, 2), dtype="float32"))


def test_view_out_computation():
    a = torch.zeros(4, device="lazy")
    b = torch.ones([2, 2], device="lazy")
    v = a.view(2, 2)
    torch.add(b, 1, out=v)
    check(a, np.ones(4, dtype="float32") * 2)
    check(v, np.ones((2, 2), dtype="float32") * 2)


def test_view_data_slice():
    t1 = torch.zeros(50, device="lazy")
    t1_slice = t1.data[:5]
    # Assigning the view back to origonal tensor's data should be OK.
    t1.data = t1_slice
    check(t1, np.zeros(5, dtype="float32"))


if __name__ == "__main__":
    pytest.main([__file__])
