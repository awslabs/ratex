# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities."""
# pylint: disable=c-extension-no-member, protected-access
import functools
import time
import traceback

import tvm

import _RATEXC


class ltc_timed:  # pylint: disable=invalid-name
    """A wrapper to add a timed sample to metric report. It can be used as a decorator or
    a context manager:

    Examples
    --------

    .. code-block:: python

        @ltc_timed("my-metric")
        def my_func():
            ...

        def my_func():
            with ltc_timed("my-metric"):
                ...
    """

    # pylint: disable=missing-docstring

    def __init__(self, name):
        self.name = name
        self.start = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        _RATEXC._raf_ltc_timed_metric(self.name, 1e9 * (time.time() - self.start))

    def __call__(self, func):
        @functools.wraps(func)
        def _timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return _timer


def ltc_counter(name, value=1):
    """A wrapper to add a counter sample to metric report."""
    _RATEXC._raf_ltc_counter_metric(name, value)


def to_torch_name(name):
    """Transform the parameter naming style to PyTorch."""
    if name.startswith("model_"):
        assert name.startswith("model_")
        name = name[len("model_") :]
        name = name.replace("_", ".")
    return name


def to_raf_name(name):
    """Transform the parameter naming style to RAF."""
    return "model_" + name.replace(".", "_")


@tvm._ffi.register_func("ratex.utils.print_stack")
def print_stack():
    """Print stack trace."""
    print("python stack trace: ")
    traceback.print_stack()
