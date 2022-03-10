#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

# apex requires a specific PyTorch version
python3 -m pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

mkdir -p build && cd build
git clone https://github.com/szhengac/apex --branch lans
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../..

