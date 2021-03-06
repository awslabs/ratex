# Copyright (c) 2018 Google Inc. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function

import _RATEXC


def counter_names():
    """Retrieves all the currently active counter names."""
    return _RATEXC._ltc_counter_names()


def counter_value(name):
    """Returns the value of an active counter.

    Args:
      name (string): The name of the counter whose value needs to be retrieved.

    Returns:
      The counter value as integer.
    """
    return _RATEXC._ltc_counter_value(name)


def metric_names():
    """Retrieves all the currently active metric names."""
    return _RATEXC._ltc_metric_names()


def metric_data(name):
    """Returns the data of an active metric.

    Args:
      name (string): The name of the metric whose data needs to be retrieved.

    Returns:
      The metric data, which is a tuple of (TOTAL_SAMPLES, ACCUMULATOR, SAMPLES).
      The `TOTAL_SAMPLES` is the total number of samples which have been posted to
      the metric. A metric retains only a given number of samples (in a circular
      buffer).
      The `ACCUMULATOR` is the sum of the samples over `TOTAL_SAMPLES`.
      The `SAMPLES` is a list of (TIME, VALUE) tuples.
    """
    return _RATEXC._ltc_metric_data(name)


def metrics_report():
    """Retrieves a string containing the full metrics and counters report."""
    return _RATEXC._ltc_metrics_report()
