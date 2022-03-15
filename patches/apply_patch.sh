#! /usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Use the specified version if specified. Otherwise, use the installed PyTorch version.
VERSION=$(python3 -c "import torch; print(torch.__version__)")
if [ "$#" -eq 1 ]; then
    VERSION=$1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAZOR_DIR=$SCRIPT_DIR/../
PYTORCH_INSTALL_PATH=$(dirname `python3 -c "import torch; print(torch.__file__)"`)

if [[ $VERSION = 1.11* ]]; then
    PATCH_DIR=$SCRIPT_DIR/torch_1.11
    pushd .
    cd $RAZOR_DIR/
    git apply --check $PATCH_DIR/razor.patch
    if [[ $? == 0 ]]; then
        git apply $PATCH_DIR/razor.patch
    fi
    popd

    patch -R -s -f --dry-run $PYTORCH_INSTALL_PATH/_tensor.py < $PATCH_DIR/torch.patch > /dev/null
    if [[ $? == 1 ]]; then
        patch $PYTORCH_INSTALL_PATH/_tensor.py < $PATCH_DIR/torch.patch
    fi
    echo "Applied patch for PyTorch $VERSION"
else
    echo "No patch for PyTorch version: $VERSION"
fi
