#!/usr/bin
# Please make sure you are at the PyTorch folder when running this script.
set -ex

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
export PYTORCH_SOURCE_PATH=`pwd`
export LTC_SOURCE_PATH=${PYTORCH_SOURCE_PATH}/lazy_tensor_core

echo "Building torch_mnm wheels..."
pushd .
cd ${PYTORCH_SOURCE_PATH}/torch_mnm
rm -rf ./build/pip/public/torch_mnm
python3 setup.py bdist_wheel -d ./build/pip/public/torch_mnm
pip3 install ./build/pip/public/torch_mnm/*.whl --force-reinstall --no-deps
popd

echo "Testing..."
pushd .
cd $HOME
python3 -c "import torch_mnm"
popd
