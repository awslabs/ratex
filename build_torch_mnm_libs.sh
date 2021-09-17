#!/usr/bin/env bash

set -ex

OPTS=()

while getopts "O:" OPTION
do
    case $OPTION in
       O)
           for i in ${OPTARG}; do
               OPTS+=("--cxxopt=${i}")
           done
           ;;
    esac
done
shift $(($OPTIND - 1))

CMD="${1:-install}"

cd "$(dirname "$0")"
PWD=`printf "%q\n" "$(pwd)"`
BASE_DIR="$PWD"
echo $BASE_DIR
THIRD_PARTY_DIR="$BASE_DIR/third_party"

MODE="opt"
if [[ "$XLA_DEBUG" == "1" ]]; then
  MODE="dbg"
fi

VERBOSE=
if [[ "$XLA_BAZEL_VERBOSE" == "1" ]]; then
  VERBOSE="-s"
fi

TPUVM_FLAG=
if [[ "$TPUVM_MODE" == "1" ]]; then
  TPUVM_FLAG="--define=with_tpu_support=true"
fi

MAX_JOBS=
if [[ "$XLA_CUDA" == "1" ]] && [[ "$CLOUD_BUILD" == "true" ]]; then
  MAX_JOBS="--jobs=16"
fi

OPTS+=(--cxxopt="-std=c++14")
if [[ $(basename -- $CC) =~ ^clang ]]; then
  OPTS+=(--cxxopt="-Wno-c++11-narrowing")
fi

if [[ "$XLA_CUDA" == "1" ]]; then
  OPTS+=(--cxxopt="-DXLA_CUDA=1")
  OPTS+=(--config=cuda)
fi

if [ "$CMD" == "clean" ]; then
  pushd $THIRD_PARTY_DIR/tensorflow
  bazel clean
  popd
else
  pushd $THIRD_PARTY_DIR/meta
  mkdir -p build
  cp cmake/config.cmake build/
  cd build
  cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug ..
  make -j16
  popd
  rm -rf torch_mnm/lib
  mkdir -p torch_mnm/lib
  chmod 0644 $THIRD_PARTY_DIR/meta/build/lib/libmnm.so
  chmod 0644 $THIRD_PARTY_DIR/meta/build/lib/libtvm.so
  chmod 0644 $THIRD_PARTY_DIR/meta/build/lib/libtvm_runtime.so
  ln -s $THIRD_PARTY_DIR/meta/build/lib/libmnm.so torch_mnm/lib/libmnm.so
  ln -s $THIRD_PARTY_DIR/meta/build/lib/libtvm.so torch_mnm/lib/libtvm.so
  ln -s $THIRD_PARTY_DIR/meta/build/lib/libtvm_runtime.so torch_mnm/lib/libtvm_runtime.so
fi
