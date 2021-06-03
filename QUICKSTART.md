# PyTorch/RAF

1. Clone a copy of the PyTorch repo, switch to the lazy_tensor_staging branch.

```
git clone git@github.com:pytorch/pytorch.git --recursive
cd pytorch
git checkout lazy_tensor_staging
git checkout 0e8776b4d45a243df6e8499d070e2df89dcad1f9
git submodule update --recursive
```

2. Clone a copy of the torch_mnm

```
git clone git@github.com:meta-project/torch_mnm.git --recursive
```

3. Build

Suggested build environment:

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

Build PyTorch:

```
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
python setup.py develop
```

Build Lazy Tensor Core:

```
cd lazy_tensor_core
scripts/apply_patches.sh
python setup.py develop
cd ..
```

Build PyTorch/RAF

```
cd torch_mnm
scripts/apply_patches.sh
python setup.py develop
cd ..
```

4. Run LeNet

```
cd torch_mnm
python lenet.py
```

Expected output (loss may vary):

```
mnm starts...
Epoch 0/1
----------
train Loss: 2.1834 Acc: 0.0000
Epoch 1/1
----------
train Loss: 2.1820 Acc: 0.0000
cpu starts...
Epoch 0/1
----------
train Loss: 2.1834 Acc: 0.0000
Epoch 1/1
----------
train Loss: 2.1820 Acc: 0.0000
```
