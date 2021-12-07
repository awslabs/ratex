"""Utilities."""
# pylint: disable=c-extension-no-member, protected-access
import functools
import time
import traceback

import tvm

from .. import _TORCHMNMC


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
        _TORCHMNMC._mnm_ltc_timed_metric(self.name, 1e9 * (time.time() - self.start))

    def __call__(self, func):
        @functools.wraps(func)
        def _timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return _timer


def ltc_counter(name, value=1):
    """A wrapper to add a counter sample to metric report."""
    _TORCHMNMC._mnm_ltc_counter_metric(name, value)


@tvm._ffi.register_func("torch_mnm.utils.print_stack")
def print_stack():
    """Print stack trace."""
    print("python stack trace: ")
    traceback.print_stack()
