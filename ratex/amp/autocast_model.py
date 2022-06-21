# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMP model."""
# pylint: disable=c-extension-no-member
import functools

import _RATEXC


class autocast:  # pylint: disable=invalid-name
    """We do not leverage PyTorch AMP because the list of supported ops is different."""

    # pylint: disable=missing-docstring

    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        _RATEXC._raf_set_amp_enabled(self._enabled)

    def __exit__(self, *args):
        _RATEXC._raf_set_amp_enabled(False)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_autocast
