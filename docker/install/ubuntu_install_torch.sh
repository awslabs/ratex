#!/usr/bin
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

# Install PyTorch nightly build
python3 -m pip install --force-reinstall --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
PYTORCH_GIT_SHA=$(python3 -c "import torch; print(torch.version.git_version)")
PYTORCH_INSTALL_PATH=$(dirname `python3 -c "import torch; print(torch.__file__)"`)

# Install libtorch with ABIs
pushd .
cd /tmp
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
cp -rf libtorch/* $PYTORCH_INSTALL_PATH/
rm -rf libtorch libtorch-cxx11-abi-shared-with-deps-latest.zip
popd

# Clone PyTorch for headers
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout $PYTORCH_GIT_SHA

python3 -m pip install -r requirements.txt

