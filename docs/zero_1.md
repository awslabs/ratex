<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# What is ZeRO-1 and How Can You Use It

In this document, we discuss ZeRO-1 and how to use it in Ratex.

As models increase in size, speeding up training becomes increasingly difficult. Typically, data-distributed parallelism (DDP) is one of the simplest ways to reduce training time through parallelization. It works by separating batches of data across multiple processors to be processed independently with gradients being summed afterwords. However, training a model like so implies that each parallel worker stores a copy of the entire model locally. Due to the memory limitations of GPU’s, this poses a strong limit on the size of a model that can be trained efficiently using DDP.

ZeRO is a form of data parallelism that works to reduce peak GPU memory consumption by partitioning model states across the parallel workers. By doing so it offers the potential to train models several magnitudes larger than with simple data distributed parallelism.

ZeRO-1 is the first stage of ZeRO, which specifically partitions the optimizer states. Below, we discuss how to use ZeRO-1 to parallely train your model through an example.

## 1. Ratex Set-Up

### Ratex already installed

If you already have Ratex and its dependencies installed, we need to rebuild RAF with the appropriate settings. Run the following commands in the Ratex directory.

```
# Remove current build
rm -rf third_party/raf/build
# Re-build RAF and other third party dependencies
BUILD_TYPE=Debug USE_NCCL=ON USE_CUDA=ON CUDA_ARCH=75 bash ./scripts/build_third_party.sh
```

### Ratex not installed

The below instructions are a more succinct form of the guide at https://github.com/awslabs/ratex/blob/main/docs/QUICKSTART.md. Note that they assume the use of a pre-built Docker container (referred to as Option 1 in Step 1 of QUICKSTART.md) for a quick installation. If you prefer to use a different option follow the relevant instructions in the guide and skip to step 1.2 below.

### 1.1 (Option 1: Docker Image Users only)

Install Docker if it is not already installed

```
# Pull the docker image
sudo docker pull metaprojdev/ratex:ci_gpu-latest
```

### 1.2

```
# Clone the Ratex repository
git clone git@github.com:awslabs/ratex.git —recursive
cd ratex/
```

### 1.3 (Option 1: Docker Image Users only)

```
# Acticate your Docker container
./docker/bash.sh metaprojdev/ratex:ci_gpu-latest
```

### 1.4

```
# Build third party dependencies
BUILD_TYPE=Debug USE_NCCL=ON USE_CUDA=ON CUDA_ARCH=75 bash ./scripts/build_third_party.sh

#Build Ratex
git config —global —add safe.directory /pytorch
bash ./scripts/build_ratex.sh
```

## 2. Run Example

We will test the installation and demonstrate how ZeRO-1 can be used

### 2.1

A sample model is provided in [ratex/docs/logistics_zero.py](https://github.com/awslabs/ratex/blob/main/docs/logistics_zero1.py), you can run the script using

```
RATEX_DEVICE=GPU ENABLE_PARAM_ALIASING=true LTC_IO_THREAD_POOL_SIZE=1 mpirun -np 2 --allow-run-as-root python3 logistics_zero1.py
```

Expected Output:

```
train Loss: 2.2748
Epoch 1/9
----------
train Loss: 2.4750
Epoch 1/9
----------
train Loss: 2.2570
Epoch 2/9
----------
train Loss: 2.2570
Epoch 2/9
----------
train Loss: 2.0060
Epoch 3/9
----------
train Loss: 2.0060
Epoch 3/9
----------
train Loss: 1.7602
Epoch 4/9
----------
train Loss: 1.7602
Epoch 4/9
----------
train Loss: 1.5353
Epoch 5/9
----------
train Loss: 1.5353
Epoch 5/9
----------
train Loss: 1.3343
Epoch 6/9
----------
train Loss: 1.3343
Epoch 6/9
----------
train Loss: 1.1585
Epoch 7/9
----------
train Loss: 1.1585
Epoch 7/9
----------
train Loss: 1.0076
Epoch 8/9
----------
train Loss: 1.0076
Epoch 8/9
----------
train Loss: 0.8788
Epoch 9/9
----------
train Loss: 0.8788
Epoch 9/9
----------
train Loss: 0.7697
train Loss: 0.7697
```

You can ignore the following warnings:

```
UserWarning: Failed to load image Python extension: /usr/local/lib/python3.7/dist-packages/torchvision/image.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs warn(f"Failed to load image Python extension: {e}")
```
```
Warning: Output.x is not binded to a call node: #[version = "0.0.5"]
```

## 3. Interface

The ZeRO-1 feature is activated by setting the configuration variable, zero_opt_level. An example can be seen in line 116 and 117 of logistics_zero.py.

```
# import statement from line 12 included for clarity
from raf import distributed as dist

dcfg = dist.get_config()
dcfg.zero_opt_level = 1
```

## 4. LTC IR FILEs

To see the effect of ZeRO-1 in the IR, we can increase the verbosity of the output.

### 4.1

Uncomment lines 26 and 27 of logistics_zero.py.

```
import _RATEXC
_RATEXC._set_ratex_vlog_level(-5)
```

### 4.2

Re-run the script, this time setting the following additional environment variables.
LTC_SAVE_TENSORS_FILE: file name to append the output to
LTC_IR_DEBUG: adds the file path and line that each operation originates from

```
LTC_SAVE_TENSORS_FILE="ltc.txt" LTC_IR_DEBUG=1 RATEX_DEVICE=GPU ENABLE_PARAM_ALIASING=true LTC_IO_THREAD_POOL_SIZE=1 mpirun -np 2 --allow-run-as-root python3 logistics_zero1.py
```

### 4.3

Open ltc.txt. Notice line 73:

```
%21 = f32[5,784] lazy_tensors::select(%20.0), location=step@sgd.py:84, dim=0, start=0, end=5, stride=1,   step (/workspace/.local/lib/python3.7/site-packages/ratex/optimizer/sgd.py:84)
```

This “select” operation represents the partition of the shards. This model shards vectors of length 10 into 2 partitions, and we see this operation is selecting the first shard.

You can also observe the sgd implementation of the operation by looking at the referenced line in the optimizer. An excerpt of surrounding lines 81-84 from [ratex/ratex/optimizer/sgd.py](https://github.com/awslabs/ratex/blob/main/ratex/optimizer/sgd.py) is provided below.

```
grad_slice = grad[self._rank * part_size : (self._rank + 1) * part_size]
data_slice = data[self._rank * part_size : (self._rank + 1) * part_size]

momentum_buffer.mul_(momentum).add_(grad_slice)
```

## Further Resources

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)