#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y build-essential
apt-get install -y python3 python3-dev python3-pip
apt-get install -y python3.7 python3.7-dev python3.7-venv
rm /usr/bin/python3
ln -s /usr/bin/python3.7 /usr/bin/python3

python3 -m pip install -U --force-reinstall pip
python3 -m pip install cmake
python3 -m pip install scikit-build==0.11.1
python3 -m pip install pylint==2.4.3 cpplint black==20.8b1
python3 -m pip install six numpy pytest cython decorator scipy tornado typed_ast \
                       pytest mypy orderedset antlr4-python3-runtime attrs requests \
                       Pillow packaging psutil dataclasses pycparser pydot filelock
python3 -m pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cffi \
                       typing_extensions future glob2 pygithub boto3
python3 -m pip install datasets==1.15.1
python3 -m pip install transformers==4.3

if [[ "$1" == "gpu" ]]; then
    python3 -m pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    mkdir -p build && cd build
    git clone https://github.com/szhengac/apex --branch lans
    cd apex
    pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    cd ../..
fi
