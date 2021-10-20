#!/bin/bash

set -ex

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR="$CDIR/.."
PTDIR=${PTDIR:="$XDIR/.."}
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PTDIR/build/aten/src/ATen"
fi

pushd $PTDIR
python -m tools.codegen.gen_backend_stubs \
  --output_dir="$XDIR/torch_mnm/csrc" \
  --source_yaml="$XDIR/mnm_native_functions.yaml"\

popd
