#!/usr/bin
# Please make sure you are at the PyTorch folder when running this script.
set -ex

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0

if [ -z $PYTORCH_SOURCE_PATH ]; then
  echo "PYTORCH_SOURCE_PATH is not set"
  exit 1
fi

pip3 install filelock

echo "Building torch_mnm wheels..."
rm -rf ./build/pip/public/torch_mnm
python3 setup.py bdist_wheel -d ./build/pip/public/torch_mnm
pip3 install ./build/pip/public/torch_mnm/*.whl --upgrade --force-reinstall --no-deps

echo "Testing..."
pushd .
cd $HOME
python3 -c "import torch_mnm; print(torch_mnm.__file__)"
popd
