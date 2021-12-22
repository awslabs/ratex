#!/usr/bin
# 1. Please make sure you are at the PyTorch folder when running this script.
# 2. Before building PyTorch, please make sure the PyTorch is configured
#    to the lazy tensor core branch:
#      git checkout lazy_tensor_staging
#      git checkout 0e8776b4d45a243df6e8499d070e2df89dcad1f9
#      git submodule update --recursive
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
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:51011"
export XLA_DEBUG=0
export XLA_CUDA=0
export FORCE_NNC=true
export TORCH_HOME="$(pwd)"

# Disable CUDA in PyTorch to reduce the build time
export USE_CUDA=0

# Enable MKL in PyTorch to accelerate graph tracing
export USE_MKL=1

# Build PyTorch
echo "Building PyTorch wheel..."
python3 setup.py bdist_wheel -d build/pip/public/pytorch
pip3 install build/pip/public/pytorch/*.whl --upgrade --force-reinstall

# Build LTC
echo "Building LazyTensorCore wheel..."
cd lazy_tensor_core
scripts/apply_patches.sh
export BUILD_CPP_TESTS=1
python3 setup.py bdist_wheel -d ../build/pip/public/lazy_tensor_core
pip3 install ../build/pip/public/lazy_tensor_core/*.whl --upgrade --force-reinstall

echo "Testing..."
pushd .
cd $HOME
python3 -c "import torch; print(torch.__file__)"
popd
