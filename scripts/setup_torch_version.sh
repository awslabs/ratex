#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup PyTorch CPU version.
# Usage: setup_torch_version.sh <nightly|pinned|version>
#   nightly: Install the latset nightly version.
#   pinned: Install the pinned nightly version specified in pinned_torch_nightly.txt
#   version: Install the assigned release version (e.g., 1.12.0).
set -e
set -o pipefail

if [ -z $PYTORCH_SOURCE_PATH ]; then
  echo "PYTORCH_SOURCE_PATH is not set"
  exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: setup_torch_version.sh <nightly|pinned|version>>"
    exit -1
fi

VERSION=$1
PLATFORM=cpu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install PyTorch
echo "Installing PyTorch version ${VERSION}"
if [ "$VERSION" == "nightly" ]; then
    # Latest nightly build
    python3 -m pip install --force-reinstall --pre torch -f https://download.pytorch.org/whl/nightly/$PLATFORM/torch_nightly.html
    LIBTORCH_LINK=https://download.pytorch.org/libtorch/nightly/$PLATFORM/libtorch-cxx11-abi-shared-with-deps-latest.zip
elif [ "$VERSION" == "pinned" ]; then
    # Pinned nightly build
    PINNED_NIGHTLY_VERSION=`cat ${SCRIPT_DIR}/pinned_torch_nightly.txt`
    echo "Pinned nightly: ${PINNED_NIGHTLY_VERSION}"
    python3 -m pip install --force-reinstall --pre torch==$PINNED_NIGHTLY_VERSION+$PLATFORM -f https://download.pytorch.org/whl/nightly/$PLATFORM/torch_nightly.html
    LIBTORCH_LINK=https://download.pytorch.org/libtorch/nightly/$PLATFORM/libtorch-cxx11-abi-shared-with-deps-$PINNED_NIGHTLY_VERSION%2B$PLATFORM.zip
else
    # Stable build
    python3 -m pip install torch==$VERSION+$PLATFORM -f https://download.pytorch.org/whl/$PLATFORM/torch_stable.html
    LIBTORCH_LINK=https://download.pytorch.org/libtorch/$PLATFORM/libtorch-cxx11-abi-shared-with-deps-$VERSION%2B$PLATFORM.zip
fi

PYTORCH_GIT_SHA=$(python3 -c "import torch; print(torch.version.git_version)")
PYTORCH_INSTALL_PATH=$(dirname `python3 -c "import torch; print(torch.__file__)"`)

# Install libtorch with cxx11 ABIs
pushd .
cd /tmp
wget -O libtorch-cxx11.zip $LIBTORCH_LINK
unzip -qq libtorch-cxx11.zip
cp -rf libtorch/* $PYTORCH_INSTALL_PATH/
rm -rf libtorch libtorch-cxx11.zip
popd

# Checkout the commit in PyTorch repo.
pushd .
cd $PYTORCH_SOURCE_PATH
git fetch
git checkout $PYTORCH_GIT_SHA
cp -r torch/csrc/distributed $PYTORCH_INSTALL_PATH/include/torch/csrc/
popd

echo "====== PyTorch Env Details ======="
echo "Version: `python3 -c \"import torch; print(torch.__version__)\"`"
echo "Git SHA: $PYTORCH_GIT_SHA"
echo "Install path: $PYTORCH_INSTALL_PATH"
echo "Source path: $PYTORCH_SOURCE_PATH"
echo "=================================="
