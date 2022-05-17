#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables you are probably interested in:
#
#   BUILD_TYPE
#     build with "Debug" or "Release" mode. Default: "Debug"
#
#   USE_CUDA
#     build with CUDA. Default: ON
#
#   CUDA_ARCH
#     the CUDA architecture (sm_xx) of the target GPU. Default: 70
#
#   USE_CUTLASS
#     build with Cutlass. Default: OFF
#
#   USE_NCCL
#     build with NCCL. Default: OFF
#
#   BUILD_MAX_JOBS
#     maximum number of jobs to build. Default: all CPU cores - 1.
set -ex

BASE_DIR=`pwd`

if [[ -z ${BUILD_TYPE+x} ]]; then
  BUILD_TYPE="Debug"
fi

if [[ -z ${USE_CUDA+x} ]]; then
  USE_CUDA="ON"
fi

if [[ -z ${CUDA_ARCH+x} ]]; then
  CUDA_ARCH="70"
fi

if [[ -z ${USE_CUTLASS+x} ]]; then
  USE_CUTLASS="OFF"
fi

if [[ -z ${USE_NCCL+x} ]]; then
  USE_NCCL="OFF"
fi

if [[ -z ${BUILD_MAX_JOBS+x} ]]; then
  BUILD_MAX_JOBS=$(expr $(nproc) - 1)
fi

USE_CUBLAS="OFF"
USE_CUDNN="OFF"
USE_MPI="OFF"
if [ "$USE_CUDA" = "ON" ]; then
  USE_CUBLAS="ON"
  USE_CUDNN="ON"
  if [ "$USE_NCCL" = "ON" ]; then
    USE_MPI="ON"
  fi
fi

echo "Building RAF/TVM..."
pushd .
cd third_party/raf/
bash ./scripts/src_codegen/run_all.sh
mkdir -p build
cp cmake/config.cmake build/
cd build
echo "set(CMAKE_BUILD_TYPE ${BUILD_TYPE})" >> config.cmake
echo "set(RAF_USE_CUDA $USE_CUDA)" >> config.cmake
echo "set(RAF_CUDA_ARCH $CUDA_ARCH)" >> config.cmake
echo "set(RAF_USE_CUBLAS $USE_CUBLAS)" >> config.cmake
echo "set(RAF_USE_CUDNN $USE_CUDNN)" >> config.cmake
echo "set(RAF_USE_CUTLASS $USE_CUTLASS)" >> config.cmake
echo "set(RAF_USE_NCCL $USE_NCCL)" >> config.cmake
echo "set(RAF_USE_MPI $USE_MPI)" >> config.cmake
cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ ..
make -j${BUILD_MAX_JOBS}
popd

echo "Building RAF/TVM wheels..."
pushd .
cd third_party/raf/3rdparty/tvm/python
rm -rf ../build/pip/public/tvm_latest
TVM_LIBRARY_PATH=${BASE_DIR}/third_party/raf/build/lib python3 setup.py bdist_wheel -d ../build/pip/public/tvm_latest
python3 -m pip install ../build/pip/public/tvm_latest/*.whl --upgrade --force-reinstall --no-deps
popd
pushd .
cd third_party/raf/python
rm -rf ../build/pip/public/raf
python3 setup.py bdist_wheel -d ../build/pip/public/raf
python3 -m pip install ../build/pip/public/raf/*.whl --upgrade --force-reinstall --no-deps
popd

echo "Testing..."
pushd .
cd $HOME
python3 -c "import tvm"
python3 -c "import raf"
popd
