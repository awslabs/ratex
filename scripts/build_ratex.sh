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

DEVELOP=0
if [ $# -eq 1 ]; then
  if [ $1 = "develop" ]; then
    DEVELOP=1
  fi
fi

if [ -z $PYTORCH_SOURCE_PATH ]; then
  echo "PYTORCH_SOURCE_PATH is not set"
  exit 1
fi

if [ $DEVELOP -eq 0 ]; then
  echo "Building ratex wheels..."
  rm -rf ./build/pip/public/ratex
  python3 setup.py bdist_wheel -d ./build/pip/public/ratex
  pip3 install ./build/pip/public/ratex/*.whl --upgrade --force-reinstall --no-deps
else
  echo "Building ratex develop..."
  python3 setup.py develop
fi

echo "Testing..."
pushd .
cd $HOME
python3 -c "import ratex; print(ratex.__file__)"
popd
