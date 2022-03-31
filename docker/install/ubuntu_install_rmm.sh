#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
tmp=$(mktemp -d)
echo "Building RAPIDSMemoryManager in directory: $tmp"

cd "$tmp"

# cmake 3.14+ required to compile RMM.
python3 -m pip install 'cmake>=3.14'

curl -sOL https://github.com/rapidsai/rmm/archive/v0.15.0.tar.gz
tar -xf v0.15.0.tar.gz

cd rmm-0.15.0
bash ./build.sh -n librmm

sudo_prefix=""
if command -v sudo &> /dev/null
then
    sudo_prefix="sudo"
fi

# Install RMM manually so sudo access is not given to build.sh (which would require pip-installed
# cmake to be on secure path)
$sudo_prefix cp build/librmm.so /usr/local/lib
$sudo_prefix cp -r include/rmm /usr/local/include/rmm
