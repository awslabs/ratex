"""RAZOR: RAf via pytorch LaZy tensOR.
Note that the import order in this file cannot be changed,
because the order of loading shared libraries matters.
"""
# pylint: disable=wrong-import-order
from . import _lib

import torch
import _LAZYC
import _TORCHMNMC
import lazy_tensor_core

from . import jit
from . import torch_parameter
from . import optimizer
