<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Debug and Profile Ratex

## Debug tips
It is always helpful to dump out more information during debug. To increase the verbose level in the program, you can set it from the python side by
```
import _RATEXC
_RATEXC._set_ratex_vlog_level(-5)
```
This follows the Caffe convention in PyTorch. There will be more information with smaller number. For example, -5 will print more messages than -3. -5 is the current highest verbose level.


You can also dump more information with different environment variables set. Some of the env vars are inherited from Lazy Tensor Core and some are added to debug RAF parts.

* LTC_SAVE_TENSORS_FILE

After setting this env var by `LTC_SAVE_TENSORS_FILE="ltc.txt"`, every new Lazy Tensor IR will be concatenated to the end of `ltc.txt` file.

In the generated `ltc.txt` file, the first information we are interested is where the graph is captured in Python. You can find this information in `TesnorsGraphInfo` section as shown below. This says the IR graph is captured by the `mark_step` called in train at lenet.py:86
```
[ScheduleSyncTensorsGraph]
TensorsGraphInfo:
  mark_step (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/ratex/lazy_tensor_core/core/lazy_model.py:720)
  train (lenet.py:86)
  main (lenet.py:153)
  <module> (lenet.py:167)
```

The Pytorch ops will be recorded during tracing and it creates a Lazy IR like this. This should follow the PyTorch program order which we executed.

```
## BEGIN_GRAPH
IR {
  %0 = f32[] lazy_tensors::device_data(), device=CPU:0
  %1 = f32[6,1,5,5] aten::expand(%0), size=(6, 1, 5, 5)
  ...
  %17 = f32[1,1,28,28] lazy_tensors::device_data(), device=CPU:0, ROOT=8
  %18 = tuple[] ratex::relay_function()
  %19 = tuple[] ratex::relay_expr(%18, %17, %16, %15, %14, %13, %12, %11, %10, %9), num_outputs=2, ROOT=11
  %20 = tuple[] ratex::relay_expr(%19.1, %8), num_outputs=9, ROOT=21
  %21 = f32[6,1,5,5] aten::mul(%20.1, %1)
  %22 = f32[6,1,5,5] aten::add(%16, %21), ROOT=0
  %23 = f32[] lazy_tensors::device_data(), device=CPU:0
  %24 = f32[16,6,5,5] aten::expand(%23), size=(16, 6, 5, 5)
  %25 = f32[16,6,5,5] aten::mul(%20.2, %24)
  %26 = f32[16,6,5,5] aten::add(%15, %25), ROOT=1
  %27 = f32[] lazy_tensors::device_data(), device=CPU:0
  ...
}
## END_GRAPH
```


* LTC_IR_DEBUG

If you want more information about where each node is created, you can run the program with `LTC_IR_DEBUG=1`, this will record the Python frame stack. After this env var is set, the `ltc.txt` will have information like below. `location=` shows the last Python code creates this node. If the node doesn't have `location` information, it is created from CPP side and you can trace it using GDB.

```
## BEGIN_GRAPH
IR {
  %0 = f32[] lazy_tensors::device_data(), location=_single_tensor_sgd@sgd.py:241, device=CPU:0,   _single_tensor_sgd (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/optim/sgd.py:241)
  %1 = f32[6,1,5,5] aten::expand(%0), location=_single_tensor_sgd@sgd.py:241, size=(6, 1, 5, 5),   _single_tensor_sgd (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/optim/sgd.py:241)
  %2 = f32[] prim::Constant(), location=backward@__init__.py:175, value=1,   backward (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/autograd/__init__.py:175)
  %3 = f32[6,1,5,5] aten::expand(%2), location=backward@__init__.py:175, size=(6, 1, 5, 5),   backward (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/autograd/__init__.py:175)
  %4 = f32[10] lazy_tensors::device_data(), location=train@lenet.py:75, device=CPU:0, ROOT=17,   train (lenet.py:75)
  %5 = f32[] prim::Constant(), location=backward@__init__.py:175, value=1,   backward (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/autograd/__init__.py:175)
  %6 = f32[] prim::Constant(), location=_make_grads@__init__.py:68, value=1,   _make_grads (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/autograd/__init__.py:68)
  %7 = f32[] aten::div(%6, %5), location=backward@__init__.py:175,   backward (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/autograd/__init__.py:175)
  %8 = f32[] aten::neg(%7), location=backward@__init__.py:175,   backward (/home/zczheng/anaconda3/envs/ratex-py37/lib/python3.7/site-packages/torch/autograd/__init__.py:175)
  ...
```


* RATEX_SAVE_IR_FILE

To dump the RATEX IR module, you can run with the env var `RATEX_SAVE_IR_FILE="raf_module.json"`. This saves the LAST raf graph into a json file. And you can load it back to a RAF module by `raf.ir.serialization.LoadJSON(module_json)`.


* RATEX_DUMP_ALIAS

The alias is an important feature in LTC design. If you want to check if the alias is setup correctly, you can set `RATEX_DUMP_ALIAS=alias.txt` and the alias will be dumped into `alias.txt`. The first column is the input id and second is output id. For example, the row `0 1` means the output1 will be the alias of input0 and they share the same memory space. If you don't see any aliases, it is possible you forgot to set `ENABLE_PARAM_ALIASING=true`.


## Profile the performance

We have several ways to debug th Ratex Performance.

* Metrics

We have metrics system in LTC. If you are interested in the execution latency of specific function, you can add `LTC_TIMED(...)` at the start of the function. Similarly, if you want to see how many times a function has been called, you can add `LTC_COUNTER(...)`. In Python, you can print these metrics by print

```
import ratex.lazy_tensor_core.debug.metrics as metrics
...
print(metrics.metrics_report())
```

The result looks like this. This can give you a rough idea about what is the current bottleneck in the program.

```
Metric: DeviceLockWait
  TotalSamples: 20
  Accumulator: 057.125us
  ValueRate: 009.684us / second
  Rate: 3.39056 / second
  Percentiles: 1%=001.712us; 5%=001.921us; 10%=002.003us; 20%=002.211us; 50%=002.882us; 80%=003.415us; 90%=004.104us; 95%=004.640us; 99%=004.640us
Metric: IrValueTensorToDataHandle
  TotalSamples: 28
  Accumulator: 003ms636.763us
  ValueRate: 453.750us / second
  Rate: 4.81841 / second
  Percentiles: 1%=013.203us; 5%=014.092us; 10%=015.991us; 20%=023.276us; 50%=057.186us; 80%=193.524us; 90%=236.051us; 95%=244.900us; 99%=257.255us
Metric: RAFCompile
  TotalSamples: 2
  Accumulator: 806ms578.778us
  ValueRate: 307ms863.165us / second
  Rate: 0.761845 / second
  Percentiles: 1%=389ms559.598us; 5%=389ms559.598us; 10%=389ms559.598us; 20%=389ms559.598us; 50%=417ms019.180us; 80%=417ms019.180us; 90%=417ms019.180us; 95%=417ms019.180us; 99%=417ms019.180us
```


* Sample based profiling

We can use BPF to generated a profile graph.

TBA https://github.com/iovisor/bpftrace