#!/usr/bin
# Please make sure you are at the PyTorch folder when running this script.
# Environment variables you are probably interested in:
#
#   BUILD_TYPE
#     build with "Debug" or "Release" mode. Default: "Debug"
#
#   USE_CUDA
#     build with CUDA. Default: ON
#
#   BUILD_MAX_JOBS
#     maximum number of jobs to build. Default: all CPU cores.
set -ex

TORCH_DIR=`pwd`

if [[ -z ${BUILD_TYPE+x} ]]; then
  BUILD_TYPE="Debug"
fi

if [[ -z ${USE_CUDA+x} ]]; then
  USE_CUDA="ON"
fi

USE_CUBLAS="OFF"
USE_CUDNN="OFF"
if [ "$USE_CUDA" = "ON" ]; then
  USE_CUBLAS="ON"
  USE_CUDNN="ON"
fi

echo "Building Meta/TVM..."
pushd .
cd ${TORCH_DIR}/torch_mnm/third_party/meta/
mkdir -p build
cp cmake/config.cmake build/
cd build
cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
      -D MNM_USE_CUDA=$USE_CUDA -D MNM_USE_CUBLAS=$USE_CUBLAS -D MNM_USE_CUDNN=$USE_CUDNN ..
make -j${BUILD_MAX_JOBS}
popd

echo "Building Meta/TVM wheels..."
pushd .
cd ${TORCH_DIR}/torch_mnm/third_party/meta/3rdparty/tvm/python
rm -rf ../build/pip/public/tvm_latest
TVM_LIBRARY_PATH=${TORCH_DIR}/torch_mnm/third_party/meta/build/lib python3 setup.py bdist_wheel -d ../build/pip/public/tvm_latest
python3 -m pip install ../build/pip/public/tvm_latest/*.whl --upgrade --force-reinstall --no-deps
popd
pushd .
cd ${TORCH_DIR}/torch_mnm/third_party/meta/python
rm -rf ../build/pip/public/mnm
python3 setup.py bdist_wheel -d ../build/pip/public/mnm
python3 -m pip install ../build/pip/public/mnm/*.whl --upgrade --force-reinstall --no-deps
popd

echo "Testing..."
pushd .
cd $HOME
python3 -c "import tvm"
python3 -c "import mnm"
popd
