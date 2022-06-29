<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Build Ratex from Source

Note that Ratex is a PyTorch extension, so the PyTorch headers and shared libraries have to be available during the compilation. Accordingly, we require to set the environment variable `PYTORCH_SOURCE_PATH` before the compilation.

In this guide, we will use the following directory organization:

```
$HOME
|- workspace
  |- pytorch
  |- ratex
```

And `PYTORCH_SOURCE_PATH` is set to `~/workspace/pytorch`.

## 1. Prepare Environment and PyTorch

### Option 1: Utilize docker container with preconfigured environment & PyTorch
This is the easiest and fastest way to configure PyTorch & Environment for Ratex. 

Install docker & clone https://github.com/awslabs/ratex

```
sudo docker pull metaprojdev/ratex:ci_gpu-latest
```

Under (`ratex/`)

This will start your docker container in interactive mode and mount your current working directory to workspace in docker container.

```
 ./docker/bash.sh metaprojdev/ratex:ci_gpu-latest
```

Once inside iteractive mode of docker container, continue to steps 2 & 3.

### Option 2 Create a Python virtual environment (optional but recommanded)

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
python3 -m pip install six numpy cython decorator scipy tornado typed_ast orderedset antlr4-python3-runtime attrs requests Pillow packaging psutil dataclasses pycparser pydot black

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

### 1.1 Install prerequisites

Install camke, ccache, clang, etc. Please refer to: [Build RAF on Conda](https://github.com/awslabs/raf/blob/main/docs/wiki/1_getting_start/Build-on-Conda.md)

### 1.2A Install PyTorch from PYPI (recommanded)

One option to prepare PyTorch is via `pip install`, so that you could save the time to build PyTorch from source. For example, we simply pip install the nightly build PyTorch:

```
python3 -m pip install --force-reinstall --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

On the other hand, the PyTorch official wheel has two issues the prevent us from directly using it to build Ratex:

1. The PyTorch wheel does not have all header files, so we still need to clone the PyTorch repo. In this way, Ratex will find the corresponding header files via `PYTORCH_SOURCE_PATH`.

  ```
  # First getting the git-hash for the PyTorch we just installed
  PYTORCH_GIT_SHA=$(python3 -c "import torch; print(torch.version.git_version)")
  # Clone PyTorch (no need to clone submodules)
  git clone git@github.com:pytorch/pytorch.git
  cd pytorch
  git checkout $PYTORCH_GIT_SHA
  python3 -m pip install -r requirements.txt
  ```

2. The PyTorch wheel was built without ABI. Since RAF is built with ABI, this results in incompatibile and compilation errors. Fortunately, PyTorch also releases libtorch with ABIs along with the nightly release, so we just need to download the corresponding libtorch and replace it.

  ```
  # First check the PyTorch is not built with ABI. You should see "False" if you install from nightly
  python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  # Then we download the libtorch with ABIs and manually install it
  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
  unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
  PYTORCH_INSTALL_PATH=$(dirname `python3 -c "import torch; print(torch.__file__)"`)
  cp -rf libtorch/* $PYTORCH_INSTALL_PATH/
  # Now we check the PyTorch again. This time you should see "True".
  python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  ```

### 1.2B Build PyTorch from source

Alternatively, you could also build PyTorch from source. We first clone the PyTorch repo. Note that we need to clone submodules for PyTorch compilation.

```
git clone git@github.com:pytorch/pytorch.git --recursive
cd pytorch
git submodule sync
git submodule update --init --recursive --jobs 0
```

Then we set some environment variables. Note that you need to make sure you are now at the PyTorch source folder (`pytorch/`).

```
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=gcc
export CXX=g++
export BUILD_CPP_TESTS=0
export DEBUG=0
export TORCH_HOME="$(pwd)"
export USE_CUDA=0 # We will use our own CUDA backend so we turn off the one in PyTorch.
```

(optional) Install MKL to facilitate graph-tracing:
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
# Enable MKL when building PyTorch
export USE_MKL=1 # To accelerate mode tracing.
```

Finally we could build PyTorch and install the wheel. Note that you need to make sure you are now at the PyTorch source folder (`pytorch/`).

```
python3 -m pip install -r requirements.txt
python3 -m pip install wheel
python3 setup.py bdist_wheel -d build/pip/public/pytorch
python3 -m pip install build/pip/public/pytorch/*.whl --force-reinstall
```

### 1.3 Test Build

```
# Switch to another place to make sure Python doesn't load torch in the current directory.
cd $HOME
# You should see the path under site-packages or dist-packages instead of the source folder.
python3 -c "import torch; print(torch.__file__)"
```

## 2. Build RAF/TVM

Now it's time to work on Ratex. Let's first clone the repo:

```
git clone git@github.com:awslabs/ratex.git --recursive
```

Since RAF and TVM do not have release wheels, we have to build them by ourselves for now.
When they are available, we should be able to simply use `pip install` to let pip download
and install their wheels.

You can directly run `bash ./scripts/build_third_party.sh` under `ratex/` to perform the following steps.
If you built PyTorch from source, remember to unset the `USE_CUDA` env variable before running `build_third_party.sh` to properly install RAF and TVM with GPU support. 

### 2.1 Compile RAF/TVM (under `ratex/`)

Note that you can also compile RAF with other configurations, such as
CUTLASS and NCCL supports. For benchmark, use `CMAKE_BUILD_TYPE=Release`.

```
cd third_party/raf/
bash ./scripts/src_codegen/run_all.sh  # run codegen
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
ratex/third_party/raf/src/impl/vm/vm.cc:270:57: error: ‘cudaStreamCaptureModeRelaxed’ was not declared in this scope
```

### 2.2 Build/Install RAF/TVM wheels (under `ratex/`)

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

### 2.3 Test Build

Again, you should see the paths under site-packages or dist-packages instead of the source folder.

```
python3 -c "import tvm"
python3 -c "import raf"
```

## 3. Build Ratex

### 3.1 Build (under `ratex/`)

You can directly run `bash ./scripts/build_ratex.sh` under `ratex/` to perform the following steps.

First make sure the environment is set:

```
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=gcc
export CXX=g++
export BUILD_CPP_TESTS=0
```

```
python3 -m pip install glob2 filelock
rm -rf ./build/pip/public/ratex
python3 setup.py bdist_wheel -d ./build/pip/public/ratex
python3 -m pip install ./build/pip/public/ratex/*.whl --force-reinstall --no-deps
```

Troubleshootings:
* If you encounter the following error, try to find `libpython3.7m.so.*` in your system. It might be just at the other place such as `/usr/lib/x86_64-linux-gnu/libpython3.7m.so.*`. Afterward, manually create symbolic links to `/usr/lib`.
```
make[2]: *** No rule to make target '/usr/lib/libpython3.6m.so', needed by 'test_ptltc'.  Stop.
```

### 3.2 Test Build

```
cd $HOME
python3 -c "import ratex"
```

## 4. Run LeNet Example

```
cd ratex/docs
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
[00:26:41] /home/ubuntu/torch-mnm-venv/pytorch/ratex/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:42] /home/ubuntu/torch-mnm-venv/pytorch/ratex/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:43] /home/ubuntu/torch-mnm-venv/pytorch/ratex/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:43] /home/ubuntu/torch-mnm-venv/pytorch/ratex/third_party/raf/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
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
RATEX_DEVICE=GPU python3 lenet.py
```

