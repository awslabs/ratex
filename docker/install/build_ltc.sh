#!/usr/bin
# 1. Please make sure you are at the PyTorch folder when running this script.
# 2. Before building LazyTensorCore, please make sure the PyTorch has been built
set -ex

TORCH_DIR=`pwd`

# Environment settings
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=1
export DEBUG=0
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:51011"
export XLA_DEBUG=0
export XLA_CUDA=0
export FORCE_NNC=true
export TORCH_HOME="$(pwd)"

# Build LTC
echo "Building LazyTensorCore wheel..."
cd lazy_tensor_core
scripts/apply_patches.sh
python3 setup.py bdist_wheel -d ../build/pip/public/lazy_tensor_core
pip3 install ../build/pip/public/lazy_tensor_core/*.whl --upgrade --force-reinstall

