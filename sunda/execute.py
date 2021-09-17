import numpy as np
import os, sys
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from torchvision.datasets import mnist
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
'''

# Notice tools.wrap_neff_pass decide the batch size from NEFF etc.
import time
import argparse

nrt_package_path = "/home/user/.local/lib/python3.6/site-packages"
# nrt_package_path = "/home/user/anaconda3/envs/torch_neuron/lib/python3.6/site-packages"
sys.path.insert(1, nrt_package_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuron
from wrap_neff_pass import NeffWrapped

print(torch_neuron)

torch_mnm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
meta_python_path = os.path.join(torch_mnm_path, "third_party/meta/python")
tvm_python_path = os.path.join(torch_mnm_path, "third_party/meta/3rdparty/tvm/python")
os.environ["TVM_LIBRARY_PATH"] = str(os.path.join(torch_mnm_path, "torch_mnm/lib"))
sys.path.insert(1, meta_python_path)
sys.path.insert(1, tvm_python_path)
import mnm
import tvm
del sys.path[1:3]
os.environ.pop("TVM_LIBRARY_PATH")

def execute(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--neff_cnt', required=True)
    args = parser.parse_args(argv)

    dirname = os.path.dirname(__file__)
    workspace = os.path.join(dirname, f"workspace_{args.neff_cnt}/")
    neffpath = os.path.join(workspace, "model.neff")
    inputspath = os.path.join(workspace, "inputs.json")
    aliaspath = os.path.join(workspace, "alias.json")
    resultspath = os.path.join(workspace, "results.json")
    with open(inputspath) as inputsfile:
        inputs = inputsfile.read()
        inputs = tvm.ir.load_json(inputs)
    with open(aliaspath) as aliasfile:
        alias = aliasfile.read()
        alias = tvm.ir.load_json(alias)
        alias = {k.value: v.value for k, v in alias.items()}
    neff = NeffWrapped(neffpath, use_dict=True, alias=alias)
    inputs = {k: v.numpy() for k, v in inputs.items()}
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    results_dict = neff(inputs)
    idx = [int(key[len("output_"):]) for key in results_dict.keys() if key.startswith("output_")]
    idx.sort()
    assert len(idx) == len(results_dict)
    results = [results_dict["output_" + str(i)] for i in idx]
    results = [mnm.array(x.detach().cpu().numpy()) for x in results]
    results = [x._ndarray__value for x in results]
    results = mnm._core.value.TupleValue(results)
    with open(resultspath, 'w') as outputsfile:
        outputsjson = mnm.ir.save_json(results)
        outputsfile.write(outputsjson)


if __name__ == "__main__":
    execute(sys.argv[1:])
