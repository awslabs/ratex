#!/bin/bash

set -ex

if [ -z $PYTORCH_SOURCE_PATH ]; then
  echo "PYTORCH_SOURCE_PATH is unset"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" ; pwd -P)"
BASE_DIR="$SCRIPT_DIR/.."
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PYTORCH_SOURCE_PATH/build/aten/src/ATen"
fi

pushd $PYTORCH_SOURCE_PATH
python3 -m tools.codegen.gen_backend_stubs \
  --output_dir="$BASE_DIR/torch_mnm/csrc" \
  --source_yaml="$BASE_DIR/mnm_native_functions.yaml"\
  --impl_path="$BASE_DIR/torch_mnm/csrc/aten_mnm_type.cpp"\

popd
