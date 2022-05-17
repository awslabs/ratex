#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

LTC_IO_THREAD_POOL_SIZE=1 mpirun -np 2 --allow-run-as-root python3 tests/python/op/test_distributed.py
LTC_IO_THREAD_POOL_SIZE=1 mpirun -np 2 --allow-run-as-root python3 tests/python/model/test_distributed_models.py
