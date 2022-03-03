#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Please make sure you are at the PyTorch folder when running this script.
set -ex

TORCH_DIR=`pwd`

# Environment settings
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
export DEBUG=0
export TORCH_HOME="$(pwd)"

# Disable CUDA in PyTorch to reduce the build time
export USE_CUDA=0

# Enable MKL in PyTorch to accelerate graph tracing
export USE_MKL=1

# Build PyTorch
pip3 install -r requirements.txt
echo "Building PyTorch wheel..."
rm -rf build/pip/public/pytorch
python3 setup.py bdist_wheel -d build/pip/public/pytorch
pip3 install build/pip/public/pytorch/*.whl --upgrade --force-reinstall

echo "Testing..."
pushd .
cd $HOME
python3 -c "import torch; print(torch.__file__)"
popd
