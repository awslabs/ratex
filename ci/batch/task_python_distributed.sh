#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

if [ -z $RATEX_DEVICE_COUNT ]; then
  echo "RATEX_DEVICE_COUNT is not set"
  exit 1
fi

NUM_WORKERS=$RATEX_DEVICE_COUNT

LTC_IO_THREAD_POOL_SIZE=1 mpirun -np $NUM_WORKERS --allow-run-as-root python3 tests/python/op/test_distributed.py
LTC_IO_THREAD_POOL_SIZE=1 mpirun -np $NUM_WORKERS --allow-run-as-root python3 tests/python/model/test_distributed_models.py
LTC_IO_THREAD_POOL_SIZE=1 mpirun -np $NUM_WORKERS --allow-run-as-root python3 tests/python/model/test_ratex_fully_sharded_data_parallel.py
