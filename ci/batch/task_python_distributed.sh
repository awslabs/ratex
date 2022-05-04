#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

#mpirun -np 2 --allow-run-as-root python3 tests/python/op/test_distributed.py