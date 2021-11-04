"""RAZOR: RAf via pytorch LaZy tensOR.
Note that the import order in this file cannot be changed,
because the order of loading shared libraries matters.
"""
# pylint: disable=wrong-import-order, useless-import-alias
from . import _lib

import torch
import _LAZYC
import _TORCHMNMC
import lazy_tensor_core

from . import jit
from . import torch_parameter
from . import optimizer

try:
    from .version import __version__ as __version__
    from .version import __mnm_gitrev__ as __mnm_gitrev__
    from .version import __torch_gitrev__ as __torch_gitrev__
except: # pylint: disable=bare-except
    __version__ = "dev"
    __mnm_gitrev__ = "unknown"
    __torch_gitrev__ = "unkonwn"
