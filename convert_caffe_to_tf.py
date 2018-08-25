import argparse
import code
import struct

import google.protobuf.text_format
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import (attr_value_pb2, graph_pb2, node_def_pb2,
                                       op_def_pb2)
from tensorflow.python.framework import tensor_shape, tensor_util

from caffe.proto import caffe_pb2

name_to_output_shape = {}
const_to_bottom = {} # Get op.inputs[0].name of Conv2D ops, used to set kernel channel dim 
consts_to_fix = set() # Names of kernel const inputs to Conv2D ops, to set shape during second pass
unsupported_caffe_types = set()
# layer_names = set() # Set of all layer names, keep track of tops
# top_names = set() # Set of all top_names, should match with at least 1 layer name. Otherwise create Identity

def gen_initial_graphdef(net):
        output_graph_def = graph_pb2.GraphDef()
        for i in range(len(net.layer)):
                layer = net.layer[i]
                if layer.type == "Input":
                        placeholder = node_def_pb2.NodeDef()
                        placeholder.op = 'Placeholder'
                        placeholder.name = layer.name
                        placeholder.attr["dtype"].type = 1
                        temp_shape = list(layer.input_param.shape[0].dim)
                        output_shape = [temp_shape[0], temp_shape[2], temp_shape[3], temp_shape[1]]
                        placeholder.attr["shape"].CopyFrom(attr_value_pb2.AttrValue(shape=tensor_shape.TensorShape(output_shape).as_proto()))
                        
                        output_graph_def.node.extend([placeholder])
                
                elif layer.type == "BatchNorm":
                        # Generate layer
                        train = False
                        is_not_training = layer.batch_norm_param.use_global_stats
                        if is_not_training != 1:
                                train = True
                        input_name = layer.bottom[0]
                        try:
                                moment = layer.batch_norm_param.moving_average_fraction
                        except:
                                moment = 0.99 # TensorFlow default
                        try:
                                eps = layer.batch_norm_param.eps
                        except:
                                eps = 0.001 # TensorFlow default
                        
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")[0]
                                tensor = op.outputs[0]
                                tf.layers.batch_normalization(tensor, momentum=moment, epsilon=eps, training=train)
                        
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()
                        new_len = len(output_graph_def.node)
                        tail_name = 'batch_normalization/FusedBatchNorm'

                        # Use Identity op to maintain layer.name in graph_def
                        connector = node_def_pb2.NodeDef()
                        connector.op = "Identity"
                        connector.name = layer.name
                        connector.attr["T"].type = 1
                        connector.input.extend([tail_name])
                        output_graph_def.node.extend([connector])

                        # code.interact(local=locals())

                elif layer.type == "Concat":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "ConcatV2"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        num_inputs = len(layer.bottom)
                        new_node.attr["N"].i = num_inputs

                        if num_inputs > 0:
                                for i in layer.bottom:
                                        new_node.input.extend([i]) 
                        # Generate axis input tensor
                        axis = node_def_pb2.NodeDef()
                        axis.op = "Const"
                        axis.name = new_node.name + "/axis"
                        axis.attr["dtype"].type = 3 # DT_INT32
                        axis.attr["value"].tensor.dtype = 3 # DT_INT32
                        tf_axis = 3 # Default tf.ConcatV2 axis is set to channels for 4-D tensor
                        # Get Caffe axis
                        try:
                                caffe_axis = layer.concat_param.axis
                        except:
                                caffe_axis = 1 # Default axis param for caffe.Concat
                        if caffe_axis != 1: # Concat along batch dim instead
                                tf_axis = 0
                        axis.attr["value"].tensor.int_val.append(tf_axis)

                        new_node.input.extend([axis.name])
                        output_graph_def.node.extend([axis])
                        output_graph_def.node.extend([new_node])

                elif layer.type == "Convolution":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Conv2D"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        try:
                                stride = list(layer.convolution_param.stride)[0]
                        except:
                                stride = 1 # Default stride for tf.Conv2D
                        stride_list = [1, stride, stride, 1]
                        new_node.attr["strides"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=stride_list))
                        try:
                                # Fails because padding default = 0, "VALID" anyways
                                if layer.convolution_param.pad[0] == 0:
                                        new_node.attr["padding"].s = "VALID".encode("utf-8")
                                else:
                                        new_node.attr["padding"].s = "SAME".encode("utf-8")
                        except:
                                new_node.attr["padding"].s = "VALID".encode("utf-8")
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])    

                        # Get bottom's output shape
                        input_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[0]], name="")[0]
                        bottom_shape = input_as_op.outputs[0].shape.as_list()
   

                        # Generate kernel node
                        kernel = node_def_pb2.NodeDef()
                        kernel.op = "Const"      
                        kernel.name = new_node.name + "/kernel"        
                        kernel.attr["dtype"].type = 1
                        kernel_shape = tensor_shape.TensorShape([layer.convolution_param.kernel_size[0],
                                                                layer.convolution_param.kernel_size[0],
                                                                bottom_shape[3],
                                                                layer.convolution_param.num_output]).as_proto()
                        kernel.attr["value"].tensor.tensor_shape.CopyFrom(kernel_shape) 
                        
                        new_node.input.extend([kernel.name])        

                        output_graph_def.node.extend([kernel])
                        output_graph_def.node.extend([new_node])
                        consts_to_fix.add(kernel.name)
                        const_to_bottom[kernel.name] = layer.bottom[0]                        
                
                elif layer.type == "Eltwise":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        try:
                                op_enum = layer.eltwise_param.operation
                        except:
                                op_enum = 1 # default is SUM
                        if op_enum == 0:
                                new_node.op = "Mul"
                        elif op_enum == 1:
                                new_node.op = "Add"
                        elif op_enum == 2:
                                new_node.op = "Max"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        num_inputs = len(layer.bottom)
                        if num_inputs > 0:
                                for i in layer.bottom:
                                        new_node.input.extend([i])
                        output_graph_def.node.extend([new_node])

                elif layer.type == "InnerProduct":
                        # Generate layer 
                        units = layer.inner_product_param.num_output
                        input_name = layer.bottom[0]
                
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]
                                tf.layers.dense(tensor, units, name='fully_connected/'+ layer.name)
                        
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()
                        new_len = len(output_graph_def.node)
                        tail_name = output_graph_def.node[new_len-1].name
                        
                        # Use Identity op to maintain layer.name in graph_def
                        connector = node_def_pb2.NodeDef()
                        connector.op = "Identity"
                        connector.name = layer.name
                        connector.attr["T"].type = 1
                        connector.input.extend([tail_name])
                        output_graph_def.node.extend([connector])

                elif layer.type == "LRN":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "LRN"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])
                        try:
                                caffe_alpha = layer.lrn_param.alpha
                        except:
                                caffe_alpha = 1 # Default alpha for tf.LRN
                        try:
                                caffe_beta = layer.lrn_param.beta
                        except:
                                caffe_beta = 0.5 # Default for tf.LRN
                        try:
                                caffe_local_size = layer.lrn_param.local_size
                        except:
                                caffe_local_size = 5
                        new_node.attr["alpha"].f = caffe_alpha
                        new_node.attr["beta"].f = caffe_beta
                        new_node.attr["depth_radius"].i = caffe_local_size
                        output_graph_def.node.extend([new_node])                
                
                elif layer.type == "Pooling":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "MaxPool" # MaxPool by default
                        if layer.pooling_param.pool == 1:
                                new_node.op = "AvgPool"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        try:
                                k_dim = layer.pooling_param.kernel_size
                        except:
                                k_dim = 1
                        kernel_shape = [1, k_dim, k_dim, 1]
                        new_node.attr["ksize"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=kernel_shape))
                        new_node.attr["padding"].s = "SAME".encode("utf-8")
                        try:
                                stride = layer.pooling_param.stride
                        except:
                                stride = 1
                        stride_list = [1, stride, stride, 1]
                        new_node.attr["strides"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=stride_list))
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]]) 
                        output_graph_def.node.extend([new_node])
                
                elif layer.type == "ReLU":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Relu"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])
                        output_graph_def.node.extend([new_node])
                
                elif layer.type == "Reshape":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Reshape"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        new_node.attr["Tshape"].type = 3 # DT_INT32
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])

                        # Get bottom's output shape
                        input_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[0]], name="")[0]
                        bottom_shape = input_as_op.outputs[0].shape.as_list()

                        # Generate shape node
                        shape_node = node_def_pb2.NodeDef()
                        shape_node.op = "Const"
                        shape_node.name = new_node.name + "/shape"
                        shape_node.attr["dtype"].type = 3 # DT_INT32
                        unsorted_caffe_shape = layer.reshape_param.shape.ListFields()[0][1]
                        # Convert NCHW caffe_shape to NHWC ordering
                        caffe_shape = [unsorted_caffe_shape[0],
                                                unsorted_caffe_shape[2],
                                                unsorted_caffe_shape[3],
                                                unsorted_caffe_shape[1]]
                        num_dims = len(caffe_shape)
                        temp_shape = []
                        for i in range(num_dims):
                                if caffe_shape[i] == 0:
                                        # Take note of NCHW ordering for caffe_shape vs NHWC for bottom_shape                                        
                                        temp_shape.append(bottom_shape[i])
                                else:
                                        temp_shape.append(caffe_shape[i])
                        shape_tuple = tuple(temp_shape)
                        pack_format = '<'+'l'*num_dims
                        shape_packed = struct.pack(pack_format, *shape_tuple)
                        shape_node.attr["value"].tensor.tensor_shape.dim.add(size=num_dims)
                        shape_node.attr["value"].tensor.dtype = 3 # DT_INT32
                        shape_node.attr["value"].tensor.tensor_content = shape_packed # Set 0's during second pass
                        new_node.input.extend([shape_node.name])

                        output_graph_def.node.extend([new_node])
                        output_graph_def.node.extend([shape_node])

                elif layer.type == "Softmax":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Softmax"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])
                        output_graph_def.node.extend([new_node])

                else:
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = 'Identity'
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])
                        # For user to keep track of unsuppported Caffe ops
                        if layer.type != "Identity":
                                unsupported_caffe_types.add(layer.type)
                        output_graph_def.node.extend([new_node])

        return output_graph_def

# Cleans invalid constant shape dimensions that were unknown during first forward pass
def second_pass_clean_graph(graph_def, name_to_output_shape, const_to_bottom):
        out_graph = graph_pb2.GraphDef()
        for node in graph_def.node:
                if node.name not in consts_to_fix:
                        new_node = node_def_pb2.NodeDef()
                        new_node.CopyFrom(node)
                else:
                        new_node = node_def_pb2.NodeDef()
                        new_node.CopyFrom(node)
                        temp = name_to_output_shape[node.name]
                        bottom_name = const_to_bottom[node.name]
                        bottom_shape = name_to_output_shape[bottom_name]
                        const_type = new_node.name.rsplit('/', 1)[1]
                        if const_type == 'kernel':
                                # Set 3rd dimension of kernel shape to bottom's channel dim
                                new_shape = tensor_shape.TensorShape([temp[0],temp[1],bottom_shape[3],temp[3]]).as_proto()
                                new_node.attr["value"].tensor.tensor_shape.CopyFrom(new_shape)
                out_graph.node.extend([new_node])
        out_graph.library.CopyFrom(graph_def.library)
        out_graph.versions.CopyFrom(graph_def.versions)

        return out_graph
                


## -------------------------------- MAIN ---------------------------------- ##
parser = argparse.ArgumentParser(description='Generates a TensorFlow model from a caffe prototxt.')
parser.add_argument('--model', required=True, help='Target Caffe model file. e.g. deploy.prototxt')
parser.add_argument('--output', default='converted_caffe_model.pb', help='Name of output TensorFlow model. Default is converted_model.pb.')
args = parser.parse_args()

print('[i] Input model:  ', args.model)
print('[i] Output: ', args.output)

net = caffe_pb2.NetParameter()
f = open(args.model, 'r')
net = google.protobuf.text_format.Merge(str(f.read()), net)

with tf.Session() as sess:
        output_graph_def = gen_initial_graphdef(net)
with tf.Graph().as_default() as graph:
	tf.import_graph_def(output_graph_def, name='')
for op in graph.get_operations():
        # print(op.name, ' type:', op.type)
        # print(op.outputs[0].shape.as_list())
        name_to_output_shape[op.name] = op.outputs[0].shape.as_list()
        # code.interact(local=locals())

# out = second_pass_clean_graph(output_graph_def, name_to_output_shape, const_to_bottom)
with open(args.output, "wb") as f:
        f.write(output_graph_def.SerializeToString())
print('Unsupported Caffe ops: ', unsupported_caffe_types)
f.close()
