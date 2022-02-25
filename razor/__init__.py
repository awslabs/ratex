# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAZOR: RAf via pytorch LaZy tensOR.
Note that the import order in this file cannot be changed,
because the order of loading shared libraries matters.
"""
# pylint: disable=wrong-import-order, useless-import-alias
from . import _lib

import torch

from . import lazy_tensor_core

from . import amp
from . import core
from . import jit
from . import torch_parameter
from . import optimizer
from .utils import cache

try:
    from .version import __version__ as __version__
    from .version import __raf_gitrev__ as __raf_gitrev__
    from .version import __torch_gitrev__ as __torch_gitrev__
except:  # pylint: disable=bare-except
    __version__ = "dev"
    __raf_gitrev__ = "unknown"
    __torch_gitrev__ = "unkonwn"
