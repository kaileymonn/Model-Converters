from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import code
import math
import os
import struct
import sys

import numpy as np
import onnx
import tensorflow as tf
from tensorflow.core.framework import (attr_value_pb2, graph_pb2, node_def_pb2,
                                       op_def_pb2)
from tensorflow.python.framework import tensor_shape, tensor_util

types_in_graph = set()
unsupported_onnx_types = set()
onnx_tensor_dtype_to_tf_dtype = {
        1: 1, # float
        2: 4, # uint8
        3: 6, # int8
        4: 17, # uint16
        5: 5, # int16
        6: 3, # int32
        7: 9, # int64
        8: 7, # string
        9: 10, # bool
        10: 14, # float16
        11: 2, # double (Not supported by tensorflow)
        12: 22, # uint32
        13: 23, # uint64
        14: 8, # complex64
        15: 18, # complex128
}

def extract_summary(graph):
        name_to_graph_input = {}
        name_to_tensor = {}
        tensors = set()
        inputs = set()
        placeholders = set()
        for tensor in graph.input:
                name_to_graph_input[tensor.name] = tensor 
                inputs.add(tensor.name)
        for tensor in graph.initializer:
                name_to_tensor[tensor.name] = tensor
                tensors.add(tensor.name)
        placeholders = inputs - tensors
        return name_to_graph_input, name_to_tensor, placeholders, tensors

def create_constants(graph_def, name_to_graph_input, name_to_tensor, placeholders, tensors):
        # Create Placeholders
        for name in placeholders:
                tensor = name_to_graph_input[name]
                placeholder = node_def_pb2.NodeDef()
                placeholder.op = 'Placeholder'
                placeholder.name = name
                elem_type = tensor.type.tensor_type.elem_type
                placeholder.attr["dtype"].type = onnx_tensor_dtype_to_tf_dtype[elem_type]
                output_shape = []
                shape_proto = tensor.type.tensor_type.shape.dim

                # Convert NCHW ordering to NHWC ordering
                if len(shape_proto) == 4:
                        output_shape = [1,1,1,1]
                        output_shape[0] = shape_proto[0].dim_value
                        output_shape[1] = shape_proto[2].dim_value
                        output_shape[2] = shape_proto[3].dim_value
                        output_shape[3] = shape_proto[1].dim_value
                else:
                        for d in shape_proto:
                                output_shape.append(d.dim_value)
                placeholder.attr["shape"].CopyFrom(attr_value_pb2.AttrValue(shape=tensor_shape.TensorShape(output_shape).as_proto()))
                graph_def.node.extend([placeholder])       
        
        # Create constants
        for name in tensors:
                tensor = name_to_tensor[name]
                const = node_def_pb2.NodeDef()
                const.op = 'Const'
                const.name = name
                onnx_dtype = tensor.data_type
                const.attr["dtype"].type = onnx_tensor_dtype_to_tf_dtype[onnx_dtype]
                unsorted_shape = list(tensor.dims)
                # Sort shape to NHWC
                if len(unsorted_shape) == 4: # Kernels
                        output_shape = [unsorted_shape[2], unsorted_shape[3], unsorted_shape[1], unsorted_shape[0]]
                else:
                        output_shape = unsorted_shape
                shape_proto = tensor_shape.TensorShape(output_shape).as_proto()
                const.attr["value"].tensor.tensor_shape.CopyFrom(shape_proto) 
                graph_def.node.extend([const])

def gen_initial_graphdef(graph):
        name_to_graph_input, name_to_tensor, placeholders, tensors = extract_summary(graph)
        output_graph_def = graph_pb2.GraphDef()
        create_constants(output_graph_def, name_to_graph_input, name_to_tensor, placeholders, tensors)

        for n in graph.node:
                if n.op_type == "Add":
                        # Generate node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Add"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        for name in n.input:
                                new_node.input.extend([name])
                        output_graph_def.node.extend([new_node])

                elif n.op_type == "BatchNormalization":
                        # Prepare attributes
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name

                        onnx_eps = 0.001
                        onnx_momentum = 0.99
                        onnx_is_test = 1
                        tf_train = False
                        for attr in n.attribute:
                                if attr.name == "epsilon":
                                        onnx_eps = attr.f 
                                elif attr.name == "is_test":
                                        onnx_is_test = attr.i 
                                elif attr.name == "momentum":
                                        onnx_momentum = attr.f

                        if onnx_is_test == 0:
                                tf_train = True
                        
                        # Generate node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "FusedBatchNorm"
                        new_node.name = output_name
                        new_node.attr["T"].type = 1
                        new_node.attr["epsilon"].f = onnx_eps
                        new_node.attr["is_training"].b = tf_train
                        for name in n.input:
                                new_node.input.extend([name])
                        output_graph_def.node.extend([new_node])

                elif n.op_type == "Conv":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Conv2D"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        new_node.input.extend([n.input[0]]) # Don't add weights/biases
                        stride_list = [1,1,1,1]
                        pad_bstring = "VALID".encode("utf-8")
                        weight_tensor = name_to_tensor[n.input[1]]
                        out_channels = weight_tensor.dims[0] 
                        in_channels = weight_tensor.dims[1]
                        kernel_shape_list = [1,1,in_channels,out_channels]
                        for attr in n.attribute:
                                if attr.name == "strides":
                                        stride_list[1] = attr.ints[0]
                                        stride_list[2] = attr.ints[1]
                                elif attr.name == "pads":
                                        for val in attr.ints:
                                                if val > 0:
                                                        pad_bstring = "SAME".encode("utf-8") 
                                elif attr.name == "kernel_shape":
                                        kernel_shape_list[0] = attr.ints[0]
                                        kernel_shape_list[1] = attr.ints[1]
                                #TODO: Dilations
                        new_node.attr["padding"].s = pad_bstring
                        new_node.attr["strides"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=stride_list))

                        # Generate kernel node
                        kernel = node_def_pb2.NodeDef()
                        kernel.op = "Const"      
                        kernel.name = new_node.name + "/kernel"        
                        kernel.attr["dtype"].type = 1
                        kernel_shape = tensor_shape.TensorShape(kernel_shape_list).as_proto()
                        kernel.attr["value"].tensor.tensor_shape.CopyFrom(kernel_shape) 
                        new_node.input.extend([kernel.name])        

                        output_graph_def.node.extend([kernel])
                        output_graph_def.node.extend([new_node])

                elif n.op_type == "Concat":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "ConcatV2"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        onnx_axis = n.attribute[0].i
                        num_inputs = len(n.input)
                        new_node.attr["N"].i = num_inputs
                        for name in n.input:
                                new_node.input.extend([name])

                        # Generate axis input tensor
                        axis = node_def_pb2.NodeDef()
                        axis.op = "Const"
                        axis.name = new_node.name + "/axis"
                        axis.attr["dtype"].type = 3 # DT_INT32
                        axis.attr["value"].tensor.dtype = 3 # DT_INT32

                        # # Take into account NCHW ordering for onnx.Concat vs NHWC for tf.Concat        
                        if onnx_axis == 0:
                                tf_axis = 0
                        else:
                                tf_axis = -1

                        axis.attr["value"].tensor.int_val.append(tf_axis)
                        new_node.input.extend([axis.name])

                        output_graph_def.node.extend([axis])
                        output_graph_def.node.extend([new_node])            

                elif n.op_type == "Constant":
                        # Generate node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Const"
                        new_node.attr["dtype"].type = 3 # DT_INT32
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        onnx_dims = n.attribute[0].t.dims[0]
                        onnx_dt = n.attribute[0].t.data_type
                        tf_dt = onnx_tensor_dtype_to_tf_dtype[onnx_dt]
                        onnx_raw_data = n.attribute[0].t.raw_data # as a byte string
                        new_node.attr["value"].tensor.dtype = tf_dt 
                        new_node.attr["value"].tensor.tensor_content = onnx_raw_data
                        new_node.attr["value"].tensor.tensor_shape.dim.add(size=onnx_dims)
                       
                        output_graph_def.node.extend([new_node])

                # This is more like reshape in tensorflow
                elif n.op_type == "Flatten": 
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name
                        input_name = n.input[0]
                        onnx_axis = 1
                        if len(n.attribute) > 0:
                                onnx_axis = n.attribute[0].i
                        
                        #Generate layer
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                input_tensor = op[0].outputs[0]
                                input_tensor_shape = tensor.shape.as_list()    
                                if onnx_axis == 1 and input_tensor_shape[0] == 1:
                                        shape_tensor = tf.constant([1, -1], name=output_name+'/Const')
                                        output_tensor = tf.reshape(input_tensor, shape_tensor, name=output_name)
                                elif onnx_axis == 1 and input_tensor_shape[0] > 1:
                                        shape_tensor = tf.constant([input_tensor_shape[0], -1], name=output_name+'/Const')
                                        output_tensor = tf.reshape(input_tensor, shape_tensor, name=output_name)
                                else:
                                        dim0 = 1
                                        for i in range(onnx_axis):
                                                dim0 = dim0*input_tensor_shape[i]
                                        shape_tensor = tf.constant([dim0, -1], name=output_name+'/Const')
                                        output_tensor = tf.reshape(input_tensor, shape_tensor, name=output_name)
                                
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                elif n.op_type == "Gemm":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "MatMul"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name

                        onnx_transA = 0
                        onnx_transB = 0
                        for attr in n.attribute:
                                if attr.name == "transA":
                                        onnx_transA = attr.i
                                elif attr.name == "transB":
                                        onnx_transB = attr.i

                        new_node.attr["T"].type = 1                                            
                        if onnx_transA != 0:
                                new_node.attr["transpose_a"].b = True
                        if onnx_transB != 0:
                                new_node.attr["transpose_b"].b = True

                        # Add inputs, ignore input C since we don't care about bias adds
                        new_node.input.extend([n.input[0]])
                        new_node.input.extend([n.input[1]])
                        output_graph_def.node.extend([new_node])

                elif n.op_type == "GlobalAveragePool":
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name
                        input_name = n.input[0]
                        
                        # Generate layer     
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]        
                                output_tensor = tf.keras.layers.GlobalAveragePooling2D()(tensor)  # Shape: [N, C]
                                # Convert to [N, 1, 1, C] as per onnx specification     
                                intermediate_tensor = tf.expand_dims(output_tensor, axis=1, name=output_name+'_1')
                                tf.expand_dims(intermediate_tensor, axis=1, name=output_name)                            
                        
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                elif n.op_type == "LRN":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "LRN"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name

                        # Intialize attrs using tf defaults
                        onnx_size = 5 
                        onnx_alpha = 1e-4
                        onnx_beta = 0.5
                        onnx_bias = 1.0

                        for attr in n.attribute:
                                if attr.name == "size":
                                        onnx_size = attr.i
                                elif attr.name == "alpha":
                                        onnx_alpha = attr.f                    
                                elif attr.name == "beta":
                                        onnx_beta = attr.f 
                                elif attr.name == "bias":
                                        onnx_bias = attr.f        

                        new_node.attr["alpha"].f = onnx_alpha
                        new_node.attr["beta"].f = onnx_beta
                        new_node.attr["depth_radius"].i = onnx_size
                        new_node.attr["bias"].f = onnx_bias
                        new_node.attr["T"].type = 1
                        new_node.input.extend([n.input[0]])

                        output_graph_def.node.extend([new_node])   

                elif n.op_type == "MaxPool" or n.op_type == "AveragePool":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        if n.op_type == "MaxPool":
                                new_node.op = "MaxPool"
                        else:
                                new_node.op = "AvgPool" 
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        new_node.input.extend([n.input[0]])
                        stride_list = [1,1,1,1]
                        pad_bstring = "VALID".encode("utf-8")
                        kernel_shape_list = [1,1,1,1]
                        pad_list = [0,0,0,0]
                        for attr in n.attribute:
                                if attr.name == "strides":
                                        stride_list[1] = attr.ints[0]
                                        stride_list[2] = attr.ints[1]
                                elif attr.name == "pads":
                                        for i,val in enumerate(attr.ints):
                                                pad_list[i] = val                                                
                                                if val > 0:
                                                        pad_bstring = "SAME".encode("utf-8") 
                                elif attr.name == "kernel_shape":
                                        kernel_shape_list[1] = attr.ints[0]
                                        kernel_shape_list[2] = attr.ints[1]
                        new_node.attr["ksize"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=kernel_shape_list))
                        new_node.attr["padding"].s = pad_bstring
                        new_node.attr["strides"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=stride_list))

                        # Clean output shape since onnx does weird things
                        input_as_op = tf.import_graph_def(output_graph_def, return_elements=[n.input[0]], name="")[0]
                        bottom_shape = input_as_op.outputs[0].shape.as_list()
                        onnx_out_spatial = bottom_shape
                        need_squeeze = False
                        squeeze_dims = []
                        if len(bottom_shape) > 2:
                                start_index = 1
                        else:
                                start_index = 0
                        for i in range(start_index, start_index + 2):
                                if start_index == 1:
                                        pad_total = pad_list[i*2 - 2] + pad_list[i*2 - 1]
                                        k_val = kernel_shape_list[i]
                                        s_val = stride_list[i]
                                else:
                                        pad_total = pad_list[i*2] + pad_list[i*2 + 1]
                                        k_val = kernel_shape_list[i-1]
                                        s_val = stride_list[i-1]
                                onnx_out_spatial[i] = math.floor((onnx_out_spatial[i] + pad_total - k_val) / (s_val + 1))
                                if onnx_out_spatial[i] == 0:
                                        need_squeeze = True
                                        squeeze_dims.append(i)
                        if need_squeeze == True:
                                original_name = new_node.name
                                new_node.name = new_node.name + '/presqueeze'
                                output_graph_def.node.extend([new_node])
                                with tf.Graph().as_default() as curr_graph:
                                        op = tf.import_graph_def(output_graph_def, return_elements=[new_node.name], name="")
                                        tensor = op[0].outputs[0]
                                        tf.squeeze(tensor, squeeze_dims, name=new_node.name + '/Squeeze')
                                
                                # Update output_graph_def
                                output_graph_def = curr_graph.as_graph_def()
                                tail_name = new_node.name + '/Squeeze'
                                
                                # Use Identity op to maintain layer.name in graph_def
                                connector = node_def_pb2.NodeDef()
                                connector.op = "Identity"
                                connector.name = original_name
                                connector.attr["T"].type = 1
                                connector.input.extend([tail_name])
                                output_graph_def.node.extend([connector])
                        else:
                                output_graph_def.node.extend([new_node])

                elif n.op_type == "Mul":
                        # Generate node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Mul"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        for name in n.input:
                                new_node.input.extend([name])
                        output_graph_def.node.extend([new_node])                        

                elif n.op_type == "Pad":
                        # Prepare attributes
                        onnx_mode = "constant".encode('utf-8') # Ignored here since output shape is not affected by mode
                        onnx_pads = [] # Onnx format: [x1_begin,x2_begin,...,x1_end,x2_end]
                        onnx_value = 0.0 # Ignored as well
                        input_name = n.input[0]
                        tf_pads = []
                        tf_mode = "CONSTANT"
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name   
                        for attr in n.attribute:
                                if attr.name == "mode":
                                        onnx_mode = attr.s
                                elif attr.name == "pads":
                                        for i in attr.ints:
                                                onnx_pads.append(i)
                                elif attr.name == "value":
                                        onnx_value = attr.f
                        rank = math.ceil(len(onnx_pads)/2) # Should be an int but just in case
                        for i in range(rank):
                                ith_pads = [onnx_pads[i],onnx_pads[i+rank]]
                                tf_pads.append(ith_pads)
                        
                        # Reorder to NHWC for tf_pads, onnx_pads is NCHW
                        if rank == 4:
                                myorder = [0,2,3,1]
                                tf_pads = [tf_pads[i] for i in myorder]
                        
                        if onnx_mode == "reflect".encode('utf-8'):
                                tf_mode = "REFLECT"

                        # Generate layer
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]
                                paddings = tf.constant(tf_pads, name=output_name+'/Const')
                                tf.pad(tensor, paddings, tf_mode, name=output_name)
                        
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                elif n.op_type == "Relu":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Relu"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        new_node.input.extend([n.input[0]])
                        output_graph_def.node.extend([new_node])

                elif n.op_type == "Reshape":
                        # Prepare attributes
                        is_reshape_1 = False
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name
                        input_name = n.input[0]
                        if len(n.input) > 1: # Onnx.Reshape-5
                                shape_name = n.input[1]
                        else:
                                print('Using a deprecated version of Reshape (Reshape-1) from ONNX operator set')
                                for attr in n.attribute:
                                        if attr.name == "shape":
                                                output_shape = list(attr.ints)
                                                is_reshape_1 = True
                                                    
                        # Generate layer     
                        with tf.Graph().as_default() as curr_graph:
                                if is_reshape_1 == False:
                                        input_list = tf.import_graph_def(output_graph_def, return_elements=[input_name, shape_name], name="")
                                        data_tensor = input_list[0].outputs[0]
                                        shape_tensor = input_list[1].outputs[0]  
                                elif is_reshape_1 == True:
                                        data_tensor = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")[0].outputs[0]
                                        shape_tensor = tf.constant(output_shape, name=output_name+'/Const')
                                output_tensor = tf.reshape(data_tensor, shape_tensor, name=output_name) 

                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                elif n.op_type == "Softmax":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Softmax"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        new_node.input.extend([n.input[0]])
                        output_graph_def.node.extend([new_node])
                
                elif n.op_type == "Sum":
                        # Generate node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "AddN"
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        num_inputs = len(n.input)
                        new_node.attr["N"].i = num_inputs
                        for name in n.input:
                                new_node.input.extend([name])
                        output_graph_def.node.extend([new_node])                

                elif n.op_type == "Transpose":
                        # Prepare attributes
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name
                        input_name = n.input[0]
                        onnx_perm = list(n.attribute[0].ints) # indices are in NCHW, convert to NHWC
                        tf_perm = []
                        if len(onnx_perm) == 4:
                              dim_map = {0: 0, 1: 3, 2: 1, 3: 2}  
                              for d in onnx_perm:
                                      tf_perm.append(dim_map[d])
                        else:
                                tf_perm = onnx_perm
                        
                        # Generate layer     
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]        
                                output_tensor = tf.transpose(tensor, perm=tf_perm, name=output_name) 

                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                elif n.op_type == "Upsample":
                        # Generate layer 
                        input_name = n.input[0]
                        onnx_mode = "nearest".encode('utf-8')
                        onnx_h_scale = 2.0
                        onnx_w_scale = 2.0
                        if n.name == "":
                                output_name = n.output[0]
                        else:
                                output_name = n.name                        
                        for attr in n.attribute:
                                if attr.name == "height_scale":
                                        onnx_h_scale = attr.f
                                elif attr.name == "mode":
                                        onnx_mode = attr.s
                                elif attr.name == "width_scale":
                                        onnx_w_scale = attr.f

                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]
                                tf_tensor_shape = tensor.shape.as_list()
                                new_dims = [1,1]
                                if len(tf_tensor_shape) == 4:
                                        new_dims[0] = tf_tensor_shape[1]*onnx_h_scale
                                        new_dims[1] = tf_tensor_shape[2]*onnx_w_scale
                                else:
                                        print('weird input case for upsampling')
                                if onnx_mode == "nearest".encode('utf-8'):
                                        size_tensor = tf.constant([int(new_dims[0]), int(new_dims[1])], name=output_name+'/Const')
                                        tf.image.resize_nearest_neighbor(tensor, size_tensor, name=output_name)
                                else:
                                        size_tensor = tf.constant([int(new_dims[0]), int(new_dims[1])], name=output_name+'/Const')
                                        tf.image.resize_bilinear(tensor, size_tensor, name=output_name)

                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                else:
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = 'Identity'
                        if n.name == "":
                                new_node.name = n.output[0]
                        else:
                                new_node.name = n.name
                        new_node.attr["T"].type = 1
                        if len(n.input) > 0:
                                new_node.input.extend([n.input[0]])

                        # For user to keep track of unsuppported onnx ops
                        if n.op_type != "Identity":
                                unsupported_onnx_types.add(n.op_type)
                        output_graph_def.node.extend([new_node])                        

        return output_graph_def

## -------------------------------- MAIN ---------------------------------- ##
parser = argparse.ArgumentParser(description='Converts an Onnx model to a TensorFlow model')
parser.add_argument('-m', '--model', required=True, help='Target Onnx model file. e.g. model.onnx')
parser.add_argument('-o', '--output', default='converted_onnx_model.pb', help='Name of output TensorFlow model. Default is converted_onnx_model.pb.')
args = parser.parse_args()

print('[i] Input model:  ', args.model)
print('[i] Output: ', args.output)

# Load ONNX model
onnx_model = onnx.load(args.model)

# Generate tf GraphDef, serialize, and write into protobuf
with tf.Session() as sess:
        out_graph = gen_initial_graphdef(onnx_model.graph)
with open(args.output, "wb") as f:
        f.write(out_graph.SerializeToString())
if len(unsupported_onnx_types) == 0:
        print('All Onnx layer types in this prototxt are supported')
else:
        print('Unsupported Onnx ops: ', unsupported_onnx_types)
f.close()
