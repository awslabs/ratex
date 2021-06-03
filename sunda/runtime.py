import uuid
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import pickle
import types

class NeuronForward(torch.nn.Module):
    def __init__(self, 
        input_names,
        input_shapes,
        input_dtypes,
        output_names,
        output_shapes,
        output_dtypes,
        neff_memory_buffer,
        keys=[],
        values=[]):

        import torch_neuron
        from torch_neuron.proto import metaneff_pb2
        super(NeuronForward, self).__init__()
        neff_uuid = str(uuid.uuid1()).encode()
        self.neff_uuid_ts = torch.tensor(np.frombuffer(neff_uuid, dtype=np.uint8))

        assert( len(neff_memory_buffer) != 0 )

        ## It looks like this is just a byte array
        self.neff_ts = torch.tensor(np.frombuffer(neff_memory_buffer, dtype=np.uint8))

        ## Put into a shape for inference over GRPC
        type_mapping = {
            'float32': metaneff_pb2.MetaTensor.DataType.FLOAT
        }

        metaneff = metaneff_pb2.MetaNeff()

        assert len(input_names) == len(input_shapes)
        assert len(input_names) == len(input_dtypes)
        assert len(output_names) == len(output_shapes)
        assert len(output_names) == len(output_dtypes)

        for name, shape, dtype in zip(input_names, input_shapes, input_dtypes):
            tensor = metaneff.input_tensors.add()
            tensor.name = str.encode(name)
            tensor.shape[:] = shape
            tensor.data_type = type_mapping[dtype]
            
        for name, shape, dtype in zip(output_names, output_shapes, output_dtypes):
            tensor = metaneff.output_tensors.add()
            tensor.name = str.encode(name)
            tensor.shape[:] = shape
            tensor.data_type = type_mapping[dtype]
            
        # output_names permutes the original output tensors in relay
        # here we remap the alias according to the permutation
        # temporarily it relies on naming convention, which is hacky
        self.keys = keys
        self.values = []
        for v in values:
            idx = output_names.index("output_" + str(v))
            self.values.append(idx)

        self.metaneff_ts = torch.tensor(np.frombuffer(metaneff.SerializeToString(), dtype=np.uint8))

        ## Extra special hackery to update the *class* method for forward
        ## Ideally this would be in __new__ but I can't get that working yet
        neuron_function_name = "forward_" + str(len(output_shapes))
        # print("Neuron function name is {}".format(neuron_function_name))#
        if hasattr( torch.ops.neuron, neuron_function_name ):
            neuron_fn = getattr( torch.ops.neuron, neuron_function_name )
            
            def forward(self, tuple_input: List[torch.Tensor]):
                list_input = []
                ## Form required by torch script - don't change!
                for i in tuple_input:
                    list_input.append( i )

                alias_ins, alias_outs = torch.jit.annotate(List[int], []), torch.jit.annotate(List[int], [])
                for x in self.keys:
                    alias_ins.append(x)
                for x in self.values:
                    alias_outs.append(x)
                return neuron_fn( 
                    list_input,
                    self.neff_uuid_ts, 
                    self.metaneff_ts, 
                    self.neff_ts,
                    alias_ins,
                    alias_outs)

            NeuronForward.forward = forward

            # Compose the name of the function we are calling (no if statement)
            #self.neuron_function_name = "forward_" + str(len(output_shapes))
            #self.full_neuron_function_name = "torch.ops.neuron." + self.neuron_function_name

            #print("Neuron function name is{}".format(self.neuron_function_name))
        #else:
        #    raise NotImplementedError("Neuron Operations Not Found")

    def forward(self, tuple_input: List[torch.Tensor]):
        raise NotImplementedError("Please use torch.neuron.runtime.create_from_neff to construct")
        #list_input = []
        ## Form required by torch script - don't change!
        #for i in tuple_input:
        #    list_input.append( i )

        #return torch.ops.neuron.forward_1( 
        #    list_input,
        #    self.neff_uuid_ts, 
        #    self.metaneff_ts, 
        #    self.neff_ts)

    def save( self, filename ):
        scripted_module = torch.jit.script( self )
        torch.jit.save( scripted_module, filename  )

    @classmethod
    def load( cls, filename ):
        obj = torch.jit.load( filename )
        # print(obj.original_name,type(obj.original_name))
        assert( obj.original_name == "NeuronForward" )
        return obj   

def load( filename ):
    return NeuronForward.load( filename )

def create_from_neff(
    input_names,
    input_shapes,
    input_dtypes,
    output_names,
    output_shapes,
    output_dtypes,
    neff_memory_buffer,
    alias={}):

    return torch.jit.script( NeuronForward(
                input_names,
                input_shapes,
                input_dtypes,
                output_names,
                output_shapes,
                output_dtypes,
                neff_memory_buffer,
                list(alias.keys()),
                list(alias.values())) )
