"""
Copyright (C) 2020, Amazon.com. All Rights Reserved
"""
import os
import sys
import argparse
import json
import io
import tarfile
import torch
import subprocess
import numpy as np
from functools import partial
from typing import Tuple, List
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, curr_path)
import runtime
#from torch_neuron import runtime

# parser = argparse.ArgumentParser(description = 'Pytorch Neff wrapper runs')
# parser.add_argument('--neff', required=True)
cwd = os.getcwd()
# .json and .neff should be loaded into this directory
neff_path = cwd + '/tools/file.neff'

class NeffWrapped():
    input_tensors = []
    non_param_names = ['x', 'y_true']

    @staticmethod
    def str2type(string):
        type_map = {'float32': torch.float32, 'int32': torch.int32, 'int64': torch.int64}
        assert(string in type_map)
        return type_map[string]

    @staticmethod
    def getNeffJson(neffPathName):
        cmd = 'dd if={} bs=1024 skip=1 | tar xfzO - neff.json'.format(neffPathName)
        jsonText = subprocess.check_output(cmd, shell=True)
        jsonData = json.loads(jsonText.decode("utf-8"))
        return jsonData

    def load_tensor(self, name):
        filename = os.path.join(self.neff_dir, "value_" + name + ".npy")
        try:
            return torch.tensor(np.load(filename))
        except:
            return None

    def __init__(self, neff_path, use_dict=False, alias={}):
        self.fake_output = os.environ.get("FAKE_OUTPUT")
        self.use_dict = use_dict
        self.neff_path = neff_path
        self.neff_dir = os.path.dirname(neff_path)
        print('loading neff: {}'.format(neff_path))
        self.neff_json = self.getNeffJson(neff_path)
        with open(neff_path, 'rb') as f:
            self.neff_bytes = f.read()

        # extracting input names, dimesion, types
        neff_inputs = self.neff_json['metadata']['signatures']['default']['inputs']
        neff_inputs = {md['id']: [name, md['dtype'], md['shape']] for name, md in neff_inputs.items()}
        neff_inputs = [neff_inputs[key] for key in sorted(neff_inputs.keys())]
        self.input_shapes = [shape.copy() for _, _, shape in neff_inputs]
        self.input_types = [typ for _, typ, _ in neff_inputs]
        self.input_dtypes = [self.str2type(typ) for _, typ, _ in neff_inputs]
        self.input_names = [name for name, _, _ in neff_inputs]
        # print('----input------')
        # print(self.input_names)
        # print(self.input_types)
        # print(self.input_shapes)

        # zero input tensors
        self.input_tensors = {} if use_dict else []
        for idx, name in enumerate(self.input_names):
            dt = self.input_dtypes[idx]
            #tensor = torch.rand(self.input_shapes[idx], dtype = dt)/100
            tensor = self.load_tensor(name)
            if tensor is None:
                tensor = torch.ones(self.input_shapes[idx], dtype = dt)
            if use_dict:
                self.input_tensors[name] = tensor
            else:
                self.input_tensors.append(tensor)
            if name == 'x':
                self.batch_size = tensor.shape[0]
        self.batch_size = 1
        # print('batch_size = {}'.format(self.batch_size))
 
        # extracting output names, dimesion, types
        output_node_names = ['sg_tonga0']
        node_name_to_node = {node['name']: node for node in self.neff_json['nodes']}
        output_nodes = [node_name_to_node[name] for name in output_node_names]
        name_to_slice = {}
        start = 0
        for node in self.neff_json['nodes']:
            num_outputs = len(node['output_names'])
            name_to_slice[node['name']] = slice(start, start + num_outputs)
            start += num_outputs
        all_dtypes = self.neff_json['attrs']['dltype'][1]
        all_shapes = self.neff_json['attrs']['shape'][1]
        self.output_names = []
        self.output_types = []
        self.output_dtypes = []
        self.output_shapes = []
        for node in output_nodes:
            self.output_names.extend(node['output_names'])
            slice_in_all = name_to_slice[node['name']]
            node_type = all_dtypes[slice_in_all]
            self.output_types.extend(node_type)
            self.output_dtypes.extend(map(self.str2type, node_type))
            self.output_shapes.extend(all_shapes[slice_in_all])
        self.output_shapes = [shape.copy() for shape in self.output_shapes]
        #output_names = [name.replace('.','') for name in output_names]
        # print('-----output-----')
        # print(self.output_names)
        # print(self.output_types)
        # print(self.output_shapes)

        self.pt_meta_flow = False
        if 'output1' in self.output_names:
            self.pt_meta_flow = True
            # print("WARNING: see output1; assume PT-Meta flow")
        self.params = []
        self.params_outidx = []
        # map output gradients to params
        for idx, name in enumerate(self.input_names):
            # apply optimizer function if output has "grad" in name
            if self.pt_meta_flow:
                output_name = "output_" + name
            else:
                output_name = name + ".grad"
            if name not in self.non_param_names and output_name in self.output_names:
                output_idx = self.output_names.index(output_name)
                if self.use_dict:
                    tensor = self.input_tensors[name]
                else:
                    tensor = self.input_tensors[idx]
                # initialize known params (except bias which is 1 dim only)
                if len(self.input_shapes[idx]) > 1:
                    torch.nn.init.kaiming_uniform_(tensor)
                param = torch.nn.parameter.Parameter(data=tensor)
                self.params.append(param)
                self.params_outidx.append(output_idx)
                assert(self.input_shapes[idx] == self.output_shapes[output_idx])
                assert(self.input_dtypes[idx] == self.output_dtypes[output_idx])
            # hack for lenet: manual mapping from output to input (overwrite the mapping extracted using simple rule)
            #self.params_outidx = [0, 2, 4, 8, 6, 12, 10, 16, 14, 1, 3, 5, 7, 9, 11, 13, 15]

        if self.fake_output:
            self.fwd = partial(self.fake_fwd, self.output_names, self.output_dtypes, self.output_shapes)
        else:
            self.fwd = runtime.create_from_neff(
                self.input_names,
                self.input_shapes,
                self.input_types,
                self.output_names,
                self.output_shapes,
                self.output_types,
                self.neff_bytes,
                alias)

    def parameters(self):
        self.params_gen = (x for x in self.params)
        return self.params_gen

    @staticmethod
    def fake_fwd(output_names, output_dtypes, output_shapes, inputs):
        output_tensors = []
        for idx, name in enumerate(output_names):
            dt = output_dtypes[idx]
            #tensor = torch.rand(output_shapes[idx], dtype=dt) / 100
            tensor = torch.ones(output_shapes[idx], dtype=dt) * 3
            output_tensors.append(tensor)
        return output_tensors

    def __call__(self, inputs): #: List[torch.Tensor]):
        if self.use_dict:
            self.input_tensors.update(inputs)
            results = self.fwd_by_name(self.input_tensors)
        else:
            assert(len(inputs) <= len(self.input_tensors))
            for idx, inp in enumerate(inputs):
                self.input_tensors[idx] = inp
            results = self.fwd(self.input_tensors)
        self.update_grads(results)
        if self.pt_meta_flow:
            self.pt_meta_assign_vars(results)     
        return results

    def fwd_by_name(self, input_dict):
        output_dict = {}
        assert(type(input_dict), dict)
        input_list = []
        for idx, name in enumerate(self.input_names):
            assert(name in input_dict)
            assert(list(input_dict[name].shape) == self.input_shapes[idx])
            input_list.append(input_dict[name])
        results = self.fwd(input_list)
        assert(len(self.output_names) == len(results))
        for idx, name in enumerate(self.output_names):
            output_dict[name] = results[idx]
        return output_dict

    def update_grads(self, outputs):
        # update gradients
        for idx, param in enumerate(self.params):
            output_idx = self.params_outidx[idx]
            if self.use_dict:
                output_name = self.output_names[output_idx]
                param.grad = outputs[output_name].clone()
            else:
                params.grad = outputs[output_idx].clone()

    def pt_meta_assign_vars(self, outputs):
        # update variables
        for idx, param in enumerate(self.params):
            output_idx = self.params_outidx[idx]
            if self.use_dict:
                output_name = self.output_names[output_idx]
                param.copy_(outputs[output_name].clone())
            else:
                params.copy_(outputs[output_idx].clone())

