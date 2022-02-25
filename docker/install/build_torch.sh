#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

git clone https://github.com/pytorch/pytorch.git --recursive
cd pytorch
git submodule sync
git submodule update --recursive

python3 -m pip install -r requirements.txt

# Environment settings
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
export DEBUG=0
export TORCH_HOME="$(pwd)"

# Disable to workaround the error as reported in
# https://github.com/pytorch/pytorch/issues/41673
#export USE_NINJA=OFF

# Disable CUDA in PyTorch
export USE_CUDA=0

# Enable MKL in PyTorch to accelerate graph tracing
export USE_MKL=1

# Build PyTorch
echo "Building PyTorch wheel..."
python3 setup.py bdist_wheel -d build/pip/public/pytorch
pip3 install build/pip/public/pytorch/*.whl --upgrade --disable-pip-version-check --force-reinstall

echo "Testing..."
pushd .
cd $HOME
python3 -c "import torch; print(torch.__file__)"
popd

