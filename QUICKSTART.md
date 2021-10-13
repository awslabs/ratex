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

### 1. Clone a copy of the PyTorch repo and switch to the lazy_tensor_staging branch.

```
git clone git@github.com:pytorch/pytorch.git --recursive
cd pytorch
git checkout lazy_tensor_staging
git checkout 0e8776b4d45a243df6e8499d070e2df89dcad1f9
git submodule update --recursive
```

### 2. Clone a copy of the torch_mnm (this repo).

Note that we are now at `pytorch/`.

```
git clone git@github.com:meta-project/torch_mnm.git --recursive
```

### 3. Create a Python virtual environment (optional).

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

### 4. Build

#### 4.1. Suggested build environment (run them under `pytorch/`)

```
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
```

#### 4.2. Install required Python packages:
```
pip3 install -r requirements.txt
```

#### 4.3. Build PyTorch (under `pytorch/`):

```
python3 setup.py develop
```

#### 4.4. Build Lazy Tensor Core (under `pytorch/lazy_tensor_core`):

```
scripts/apply_patches.sh
# override BUILD_CPP_TESTS, since ltc does not work when it is 0. This is a workaround.
export BUILD_CPP_TESTS=1
python3 setup.py develop
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

#### 4.5. Build PyTorch/RAF

```
cd torch_mnm
scripts/apply_patches.sh
# it may fail when building test for now.
export BUILD_CPP_TESTS=0
python3 setup.py develop
cd ..
```

#### 4.6 Test
```
python3
>>> import torch_mnm
```

4. Run LeNet

```
cd torch_mnm
pip3 install torchvision --no-deps # otherwise it will install PyTorch from wheel...
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

