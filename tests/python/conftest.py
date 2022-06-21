# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=protected-access, c-extension-no-member, unused-import, wrong-import-order
import pytest

import ratex
import _RATEXC
from ratex.jit.script import JIT_CACHE


@pytest.fixture(autouse=True)
def reset_ratex_jit_cache():
    """ Reset the JIT cache before each pytest run."""
    JIT_CACHE.clear()
    _RATEXC._ltc_clear_jit_cache()
    yield
