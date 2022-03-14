#! /usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to update system PyTorch to the latest nightly build.
# If PYTORCH_SOURCE_PATH is specified, then this script also checks out
# the corresponding commit.
# NOTE: It is highly recommended to run this script in Conda or Docker container.
set -ex

python3 -m pip install --force-reinstall --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
PYTORCH_GIT_SHA=$(python3 -c "import torch; print(torch.version.git_version)")
PYTORCH_INSTALL_PATH=$(dirname `python3 -c "import torch; print(torch.__file__)"`)
echo "PyTorch commit: $PYTORCH_GIT_SHA"

pushd .
cd /tmp
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
cp -rf libtorch/* $PYTORCH_INSTALL_PATH/
rm -rf libtorch libtorch-cxx11-abi-shared-with-deps-latest.zip
popd

if [ ! -z $PYTORCH_SOURCE_PATH ]; then
    echo "Checkout the commit in $PYTORCH_SOURCE_PATH"
    pushd .
    cd $PYTORCH_SOURCE_PATH
    git fetch
    git checkout $PYTORCH_GIT_SHA
    popd
fi
