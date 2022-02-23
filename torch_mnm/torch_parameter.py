# pylint: disable=protected-access, c-extension-no-member
"""Hooks to PyTorch."""
import torch

import _TORCHMNMC


def _to(self, *args, **kwargs):
    ret = super(torch.nn.parameter.Parameter, self).to(*args, **kwargs)
    if str(ret.device.type) == "lazy":
        return _TORCHMNMC._mnm_mark_parameter(ret)
    return ret


torch.nn.parameter.Parameter.to = _to
