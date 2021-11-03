# PyTorch/RAF

Note that PyTorch/RAF is a PyTorch dialect based on lazy tensor core, so similar to PyTorch/XLA,
it has to be put along with the PyTorch source. Specifically, the directory organization
looks like the following:
```
pytorch
|- torch
|- ...
|- lazy_tensor_core # Only available in PyTorch lazy_tensor_staging branch (see step 1).
|- torch_mnm        # Manually clone and put here
```

### 1. Preparation

#### 1.1 Clone a copy of the PyTorch repo and switch to the lazy_tensor_staging branch

```
git clone git@github.com:pytorch/pytorch.git --recursive
cd pytorch
git checkout lazy_tensor_staging
git checkout 0e8776b4d45a243df6e8499d070e2df89dcad1f9
git submodule update --recursive
```

#### 1.2 Clone a copy of the torch_mnm (this repo)

Note that we are now at `pytorch/`.

```
git clone git@github.com:meta-project/torch_mnm.git --recursive
```

#### 1.3 Create a Python virtual environment (optional but recommanded)

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

Please refer to https://github.com/meta-project/meta/blob/master/docs/wiki/Recommended-Development-Environment.md.

### 4. Build PyTorch and LazyTensorCore

Note that you need to make sure you are now at the PyTorch source folder (`pytorch/`).
Then, you can directly run `bash torch_mnm/scripts/build_torch.sh` that performs all the following steps.
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
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:51011"
export XLA_DEBUG=0
export XLA_CUDA=0
export FORCE_NNC=true
export TORCH_HOME="$(pwd)"
export USE_CUDA=0 # We will use our own CUDA backend so we turn off the one in PyTorch.
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

#### 4.4. Build Lazy Tensor Core and install wheel (under `pytorch/lazy_tensor_core`)

```
scripts/apply_patches.sh
# override BUILD_CPP_TESTS, since ltc does not work when it is 0. This is a workaround.
export BUILD_CPP_TESTS=1
python3 -m pip install glob2
python3 setup.py bdist_wheel -d ../build/pip/public/lazy_tensor_core
python3 -m pip install ../build/pip/public/lazy_tensor_core/*.whl --force-reinstall
```

Troubleshootings:
* If you don't see `lazy_tensor_core` under `pytorch/`, check the Step 1 again to see if you have switched to `lazy_tensor_staging` branch.
* If you encounter the following error, try to find `libpython3.6m.so.*` in your system. It might be just at the other place such as `/usr/lib/x86_64-linux-gnu/libpython3.6m.so.*`. Afterward, manually create symbolic links to `/usr/lib`.
```
make[2]: *** No rule to make target '/usr/lib/libpython3.6m.so', needed by 'test_ptltc'.  Stop.
```
* If you encounter the following error, check your build environment in the beginning of this step, amd make sure using `clang`.
```
In file included from /usr/include/c++/7/list:63:0,
                 from /home/ubuntu/torch-mnm-venv/pytorch/lazy_tensor_core/lazy_tensors/computation_client/cache.h:5,
                 from /home/ubuntu/torch-mnm-venv/pytorch/lazy_tensor_core/test/cpp/test_ltc_util_cache.cpp:5:
/usr/include/c++/7/bits/stl_list.h:326:27: error: #if with no expression
 #if _GLIBCXX_USE_CXX11_ABI
```

#### 4.5 Test Build

```
# Switch to another place to make sure Python doesn't load torch in the current directory.
cd $HOME
# You should see the path under site-packages or dist-packages instead of the source folder.
python3 -c "import torch; print(torch.__file__)" 
```

### 5. Build Meta/TVM

Since Meta and TVM do not have release wheels, we have to build them by ourselves for now.
When they are available, we should be able to simply use `pip install` to let pip download
and install their wheels.

Same as building PyTorch, you can directly run `bash torch_mnm/scripts/build_third_party.sh`
under `pytorch/` to perform the following steps.

#### 5.1 Compile Meta/TVM (under `pytorch/`)

Note that you can also compile Meat with other configurations, such as
CUTLASS and NCCL supports.

```
cd torch_mnm/third_party/meta/
mkdir -p build
cp cmake/config.cmake build/
cd build
cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -D CMAKE_BUILD_TYPE=Debug \
      -D MNM_USE_CUDA=ON -D MNM_USE_CUBLAS=ON -D MNM_USE_CUDNN=ON ..
make -j$(nproc)
```

Troubleshootings:

* If you encounter the following error, try link /usr/local/cuda to /usr/local/cuda-10.2 instead of /usr/local/cuda-10.0 .
```
pytorch/torch_mnm/third_party/meta/src/impl/vm/vm.cc:270:57: error: ‘cudaStreamCaptureModeRelaxed’ was not declared in this scope
```

#### 5.2 Build/Install Meta/TVM wheels (under `pytorch/`)

```
export TORCH_DIR=`pwd`
cd torch_mnm/third_party/meta/3rdparty/tvm/python
rm -rf ../build/pip/public/tvm_latest
TVM_LIBRARY_PATH=${TORCH_DIR}/torch_mnm/third_party/meta/build/lib python3 setup.py bdist_wheel -d ../build/pip/public/tvm_latest
python3 -m pip install ../build/pip/public/tvm_latest/*.whl --force-reinstall --no-deps

cd ${TORCH_DIR}/torch_mnm/third_party/meta/python
rm -rf ../build/pip/public/mnm
python3 -m pip install decorator attrs scipy cloudpickle synr==0.5.0 tornado
python3 setup.py bdist_wheel -d ../build/pip/public/mnm
python3 -m pip install ../build/pip/public/mnm/*.whl --force-reinstall --no-deps
```

#### 5.3 Test Build

Again, you should see the paths under site-packages or dist-packages instead of the source folder.

```
python3 -c "import tvm"
python3 -c "import mnm"
```

### 6. Build PyTorch/RAF

#### 6.1 Build (under `pytorch/`)

Same as building PyTorch, you can directly run `bash torch_mnm/scripts/build_torch_mnm.sh`
under `pytorch/` to perform the following steps.

First make sure the environment is set:

```
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
export PYTORCH_SOURCE_PATH=`pwd`
export LTC_SOURCE_PATH=${PYTORCH_SOURCE_PATH}/lazy_tensor_core
```

```
cd ${PYTORCH_SOURCE_PATH}/torch_mnm
rm -rf ./build/pip/public/torch_mnm
python3 setup.py bdist_wheel -d ./build/pip/public/torch_mnm
python3 -m pip install ./build/pip/public/torch_mnm/*.whl --force-reinstall --no-deps
```

#### 6.2 Test Build

```
cd $HOME
python3 -c "import torch_mnm"
```

### 7. Run LeNet Example

```
cd torch_mnm
python3 -m pip install torchvision --no-deps # otherwise it will install PyTorch from wheel...
python3 -m pip install Pillow
python3 lenet.py
```

Expected output (loss may vary):

```
mnm starts...
Epoch 0/9
----------
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
[00:26:41] /home/ubuntu/torch-mnm-venv/pytorch/torch_mnm/third_party/meta/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:42] /home/ubuntu/torch-mnm-venv/pytorch/torch_mnm/third_party/meta/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:43] /home/ubuntu/torch-mnm-venv/pytorch/torch_mnm/third_party/meta/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
[00:26:43] /home/ubuntu/torch-mnm-venv/pytorch/torch_mnm/third_party/meta/3rdparty/tvm/src/te/autodiff/adjoint.cc:148: Warning: te.Gradient is an experimental feature.
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

