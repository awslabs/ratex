#!/usr/bin/env bash
# Environment variables you are probably interested in:
#
#   RAZOR_BUILD_TYPE
#     build with "Debug" or "Release" mode. Default: "Debug"
#
#   RAZOR_USE_CUDA
#     build with CUDA. Default: ON
#
#   RAZOR_BUILD_MAX_JOBS
#     maximum number of jobs to build. Default: all CPU cores.


set -ex

cd "$(dirname "$0")"
PWD=`printf "%q\n" "$(pwd)"`
BASE_DIR="$PWD"
echo $BASE_DIR
THIRD_PARTY_DIR="$BASE_DIR/third_party"

BUILD_TYPE=$RAZOR_BUILD_TYPE
if [[ -z ${RAZOR_BUILD_TYPE+x} ]]; then
  BUILD_TYPE="Debug"
fi

USE_CUDA=$RAZOR_USE_CUDA
if [[ -z ${RAZOR_USE_CUDA+x} ]]; then
  USE_CUDA="ON"
fi

pushd $THIRD_PARTY_DIR/meta
mkdir -p build
cp cmake/config.cmake build/
cd build
cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DMNM_USE_CUDA=$USE_CUDA ..
make -j${RAZOR_BUILD_MAX_JOBS}
popd
rm -rf torch_mnm/lib
mkdir -p torch_mnm/lib
chmod 0644 $THIRD_PARTY_DIR/meta/build/lib/libmnm.so
chmod 0644 $THIRD_PARTY_DIR/meta/build/lib/libtvm.so
chmod 0644 $THIRD_PARTY_DIR/meta/build/lib/libtvm_runtime.so
ln -s $THIRD_PARTY_DIR/meta/build/lib/libmnm.so torch_mnm/lib/libmnm.so
ln -s $THIRD_PARTY_DIR/meta/build/lib/libtvm.so torch_mnm/lib/libtvm.so
ln -s $THIRD_PARTY_DIR/meta/build/lib/libtvm_runtime.so torch_mnm/lib/libtvm_runtime.so

