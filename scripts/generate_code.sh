#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


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
python3 -m torchgen.gen_backend_stubs \
  --output_dir="$BASE_DIR/ratex/csrc" \
  --source_yaml="$BASE_DIR/raf_native_functions.yaml"\
  --impl_path="$BASE_DIR/ratex/csrc/aten_raf_type.cpp"\

popd
