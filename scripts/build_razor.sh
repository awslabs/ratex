#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=gcc
export CXX=g++
export BUILD_CPP_TESTS=0

if [ -z $PYTORCH_SOURCE_PATH ]; then
  echo "PYTORCH_SOURCE_PATH is not set"
  exit 1
fi

echo "Applying PyTorch patch..."
bash ./patches/apply_patch.sh

echo "Building razor wheels..."
rm -rf ./build/pip/public/razor
python3 setup.py bdist_wheel -d ./build/pip/public/razor
pip3 install ./build/pip/public/razor/*.whl --upgrade --force-reinstall --no-deps

echo "Testing..."
pushd .
cd $HOME
python3 -c "import razor; print(razor.__file__)"
popd
