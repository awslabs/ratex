<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# PyTorch/RAF

Note that PyTorch/RAF is a PyTorch extension, so the PyTorch source code has to be available during the compilation. Accordingly, we require to set the environment variable `PYTORCH_SOURCE_PATH` before the compilation.

In this guide, we will use the following directory organization:

```
$HOME
|- workspace
  |- pytorch
  |- razor
```

And `PYTORCH_SOURCE_PATH` is set to `~/workspace/pytorch`.

### 1. Preparation

#### 1.0 Install prerequisites

Install camke, ccache, clang, etc. Please refer to: https://github.com/meta-project/meta/blob/main/docs/wiki/1_getting_start/Build-on-Conda.md

#### 1.1 Clone a copy of the PyTorch repo

```
git clone git@github.com:pytorch/pytorch.git --recursive
cd pytorch
git submodule sync
git submodule update --init --recursive --jobs 0
```

#### 1.2 Clone a copy of the razor (this repo)

```
git clone git@github.com:meta-project/razor.git --recursive
```

#### 1.3 Create a Python virtual environment (optional but recommanded)

Note that PyTorch now requires Python 3.7+. If your system has Python 3.7-, it is recommended to use Conda.

This is recommended if you already have a PyTorch deployed and you don't want to
mess up your current PyTorch use cases.

* Option 1: virtualenv

```
mkdir -p torch-mnm-venv
cd torch-mnm-venv
python3 -m venv env
source env/bin/activate
```

* Option 2: conda

```
conda create --name py37_torch python=3.7
conda activate py37_torch
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda113

python3 -m pip install scikit-build==0.11.1
python3 -m pip install six numpy cython decorator scipy tornado typed_ast orderedset antlr4-python3-runtime attrs requests Pillow packaging psutil dataclasses pycparser pydot

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

### 4. Build PyTorch

Note that you need to make sure you are now at the PyTorch source folder (`pytorch/`).
You can directly run `bash $RAZOR_HOME/scripts/build_torch.sh` that performs all the following steps.
If you are not interested in details, you can directly jump to the last step of this section
to test whether the installation was successed.

#### 4.1. Suggested build environment

```
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
export DEBUG=0
export TORCH_HOME="$(pwd)"
export USE_CUDA=0 # We will use our own CUDA backend so we turn off the one in PyTorch.
export USE_MKL=1 # To accelerate mode tracing.
```

##### Install MKL
```
# Add GPG key
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# Add to list
wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
# Update
apt-get update
# Install. Make sure to install 2020 or later version!
apt install -y intel-mkl-2020.0-088
```

#### 4.2. Install required Python packages (under `pytorch/`)
```
python3 -m pip install -r requirements.txt
```

#### 4.3. Build PyTorch and install wheel (under `pytorch/`)

```
python3 -m pip install wheel
python3 setup.py bdist_wheel -d build/pip/public/pytorch
python3 -m pip install build/pip/public/pytorch/*.whl --force-reinstall
```

#### 4.5 Test Build

```
# Switch to another place to make sure Python doesn't load torch in the current directory.
cd $HOME
# You should see the path under site-packages or dist-packages instead of the source folder.
python3 -c "import torch; print(torch.__file__)"
```

### 5. Build RAF/TVM

Since RAF and TVM do not have release wheels, we have to build them by ourselves for now.
When they are available, we should be able to simply use `pip install` to let pip download
and install their wheels.

Same as building PyTorch, you can directly run `bash ./scripts/build_third_party.sh`
under `razor/` to perform the following steps.

#### 5.1 Compile RAF/TVM (under `razor/`)

Note that you can also compile RAF with other configurations, such as
CUTLASS and NCCL supports. For benchmark, use `CMAKE_BUILD_TYPE=Release`.

```
cd third_party/raf/
mkdir -p build
cp cmake/config.cmake build/
cd build
echo "set(CMAKE_BUILD_TYPE Debug)" >> config.cmake
echo "set(RAF_USE_CUDA ON)" >> config.cmake
echo "set(RAF_CUDA_ARCH 70)" >> config.cmake
echo "set(RAF_USE_CUBLAS ON)" >> config.cmake
echo "set(RAF_USE_CUDNN ON)" >> config.cmake
echo "set(RAF_USE_CUTLASS ON)" >> config.cmake
cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ ..
make -j$(nproc)
```

Troubleshootings:

* If you encounter the following error, try link /usr/local/cuda to /usr/local/cuda-10.2 instead of /usr/local/cuda-10.0 .
```
razor/third_party/raf/src/impl/vm/vm.cc:270:57: error: ‘cudaStreamCaptureModeRelaxed’ was not declared in this scope
```

#### 5.2 Build/Install Meta/TVM wheels (under `razor/`)

```
export BASE_DIR=`pwd`
cd third_party/raf/3rdparty/tvm/python
rm -rf ../build/pip/public/tvm_latest
TVM_LIBRARY_PATH=${BASE_DIR}/third_party/raf/build/lib python3 setup.py bdist_wheel -d ../build/pip/public/tvm_latest
python3 -m pip install ../build/pip/public/tvm_latest/*.whl --upgrade --force-reinstall --no-deps
cd -

cd third_party/raf/python
rm -rf ../build/pip/public/raf
python3 setup.py bdist_wheel -d ../build/pip/public/raf
python3 -m pip install ../build/pip/public/raf/*.whl --upgrade --force-reinstall --no-deps
```

#### 5.3 Test Build

Again, you should see the paths under site-packages or dist-packages instead of the source folder.

```
python3 -c "import tvm"
python3 -c "import raf"
```

### 6. Build PyTorch/RAF

#### 6.1 Build (under `razor/`)

Same as building PyTorch, you can directly run `bash ./scripts/build_torch_mnm.sh`
under `razor/` to perform the following steps.

First make sure the environment is set:

```
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
```

```
python3 -m pip install glob2 filelock
rm -rf ./build/pip/public/razor
python3 setup.py bdist_wheel -d ./build/pip/public/razor
python3 -m pip install ./build/pip/public/razor/*.whl --force-reinstall --no-deps
```

Troubleshootings:
* If you encounter the following error, try to find `libpython3.6m.so.*` in your system. It might be just at the other place such as `/usr/lib/x86_64-linux-gnu/libpython3.6m.so.*`. Afterward, manually create symbolic links to `/usr/lib`.
```
make[2]: *** No rule to make target '/usr/lib/libpython3.6m.so', needed by 'test_ptltc'.  Stop.
```
* If you encounter the following error, check your build environment in the beginning of this step, amd make sure using `clang`.
```
In file included from /usr/include/c++/7/list:63:0,
                 from /home/ubuntu/torch-mnm-venv/razor/lazy_tensor_core/lazy_tensors/computation_client/cache.h:5,
                 from /home/ubuntu/torch-mnm-venv/razor/lazy_tensor_core/test/cpp/test_ltc_util_cache.cpp:5:
/usr/include/c++/7/bits/stl_list.h:326:27: error: #if with no expression
 #if _GLIBCXX_USE_CXX11_ABI
```

#### 6.2 Test Build

```
cd $HOME
python3 -c "import razor"
```

### 7. Run LeNet Example

```
cd razor/docs/wiki
python3 -m pip install torchvision --no-deps # otherwise it will install PyTorch from wheel...
python3 -m pip install Pillow
python3 lenet.py
```

Expected output (loss may vary):

```
raf starts...
Epoch 0/9
----------
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
[00:26:41] /home/ubuntu/torch-mnm-venv/pytorch/razor/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:42] /home/ubuntu/torch-mnm-venv/pytorch/razor/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:43] /home/ubuntu/torch-mnm-venv/pytorch/razor/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:43] /home/ubuntu/torch-mnm-venv/pytorch/razor/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
train Loss: 0.0701
Epoch 1/9
----------
train Loss: 0.0683
Epoch 2/9
----------
train Loss: 0.0665
Epoch 3/9
----------
train Loss: 0.0647
Epoch 4/9
----------
...
```

In addition, you can also run models on GPU as follows:

```
RAZOR_DEVICE=GPU python3 lenet.py
```
