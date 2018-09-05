from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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

unsupported_caffe_types = set()

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
                        # Prepare attributes
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
                        
                        # Generate layer
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")[0]
                                tensor = op.outputs[0]
                                output_tensor = tf.layers.batch_normalization(tensor, momentum=moment, epsilon=eps, training=train, name=layer.name+'/BatchNorm')
                                tf.identity(output_tensor, name=layer.name)
     
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

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

                        # Get Caffe axis
                        try:
                                caffe_axis = layer.concat_param.axis
                        except:
                                caffe_axis = 1 # Default axis param for caffe.Concat (Channels dimension)
                        
                        # Get bottom's output shape
                        input_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[0]], name="")[0]
                        bottom_shape = input_as_op.outputs[0].shape.as_list()

                        # Take into account NCHW ordering for Caffe.Concat vs NHWC for tf.Concat        
                        if caffe_axis == 0:
                                tf_axis = 0
                        else:
                                tf_axis = -1

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
                        # new_node.attr["padding"].s = "VALID".encode("utf-8")
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
                
                elif layer.type == "Crop":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "ResizeBilinear"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1
                        new_node.attr["align_corners"].b = False
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])    

                        # Get bottom1's output shape (height and width only, generic case)
                        input1_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[0]], name="")[0]
                        bottom1_shape = input1_as_op.outputs[0].shape.as_list()

                        # Get bottom2's output shape (height and width only, generic case)
                        input2_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[1]], name="")[0]
                        bottom2_shape = input2_as_op.outputs[0].shape.as_list()
                        hw_list = bottom2_shape[1:3]

                        if hw_list == [None, None]:
                                hw_list = [-1, -1]

                        shape_tuple = tuple(hw_list)
                        pack_format = '<'+'l'*2                  

                        # Generate size node
                        size_node = node_def_pb2.NodeDef()
                        size_node.op = "Const"
                        size_node.name = new_node.name + "/size"
                        size_node.attr["dtype"].type = 3
                        size_packed = struct.pack(pack_format, *shape_tuple)
                        size_node.attr["value"].tensor.tensor_shape.dim.add(size=2)
                        size_node.attr["value"].tensor.dtype = 3 # DT_INT32
                        size_node.attr["value"].tensor.tensor_content = size_packed # Set 0's during second pass
                        new_node.input.extend([size_node.name])

                        output_graph_def.node.extend([new_node])
                        output_graph_def.node.extend([size_node])

                elif layer.type == "Deconvolution":
                        # Generate conv2D transpose
                        kernel_size = layer.convolution_param.kernel_size[0]
                        num_output = layer.convolution_param.num_output
                        try:
                                stride = list(layer.convolution_param.stride)[0]
                        except:
                                stride = 1 # Default stride for tf.Conv2D
                        input_name = layer.bottom[0]

                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]
                                output_tensor = tf.layers.conv2d_transpose(tensor, num_output, kernel_size, strides=stride, name=layer.name+'Deconvolution')
                                tf.identity(output_tensor, name=layer.name)
                                           
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

                elif layer.type == "Eltwise":
                        # Generate main node
                        new_node = node_def_pb2.NodeDef()
                        new_node.name = layer.name
                        num_inputs = len(layer.bottom)
                        try:
                                op_enum = layer.eltwise_param.operation
                        except:
                                op_enum = 1 # default is SUM
                        if op_enum == 0:
                                new_node.op = "Mul"
                        elif op_enum == 1:
                                new_node.op = "AddN"
                                new_node.attr["N"].i = num_inputs
                        elif op_enum == 2:
                                new_node.op = "Max"
                        new_node.attr["T"].type = 1
                        if num_inputs > 0:
                                for i in layer.bottom:
                                        new_node.input.extend([i])
                        output_graph_def.node.extend([new_node])

                elif layer.type == "Flatten":
                        # Generally used to flatten NHWC 4D tensor to N(H*W*C) 2D tensor
                        # Generate main node, we use a specific configuration of Reshape
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Reshape"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1 # DT_FLOAT
                        new_node.attr["Tshape"].type = 3 # DT_INT32
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])
                        try:
                                caffe_axis = layer.flatten_param.axis
                        except: 
                                caffe_axis = 1 # Default caffe value
                        try:
                                caffe_end_axis = layer.flatten_param.end_axis
                        except:
                                caffe_end_axis = -1 # Default caffe value

                        # Get bottom's output shape
                        input_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[0]], name="")[0]
                        bottom_shape = input_as_op.outputs[0].shape.as_list()     

                        # General case
                        if caffe_axis == 1 and caffe_end_axis == -1:
                                num_dims = 2
                                out_dim = np.prod(bottom_shape[1:])
                                out_shape = [-1, out_dim]
                        else: 
                                print("Unsupported non-generic case for flatten. Please review.")
                                import code
                                code.interact(local=locals())

                        # Generate shape node
                        shape_node = node_def_pb2.NodeDef()
                        shape_node.op = "Const"
                        shape_node.name = new_node.name + "/shape"
                        shape_node.attr["dtype"].type = 3 # DT_INT32
                        shape_tuple = tuple(out_shape)
                        pack_format = '<'+'l'*num_dims
                        shape_packed = struct.pack(pack_format, *shape_tuple)
                        shape_node.attr["value"].tensor.tensor_shape.dim.add(size=num_dims)
                        shape_node.attr["value"].tensor.dtype = 3 # DT_INT32
                        shape_node.attr["value"].tensor.tensor_content = shape_packed
                        new_node.input.extend([shape_node.name])

                        output_graph_def.node.extend([new_node])
                        output_graph_def.node.extend([shape_node])                       

                elif layer.type == "InnerProduct":
                        # Generate layer 
                        num_output = layer.inner_product_param.num_output
                        input_name = layer.bottom[0]
                        with tf.Graph().as_default() as curr_graph:
                                op = tf.import_graph_def(output_graph_def, return_elements=[input_name], name="")
                                tensor = op[0].outputs[0]
                                output_tensor = tf.contrib.layers.fully_connected(tensor, num_output)
                                # Create connector to match output name with layer.name
                                tf.identity(output_tensor, name=layer.name)
                        
                        # Update output_graph_def
                        output_graph_def = curr_graph.as_graph_def()

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
                        try:
                                # Fails because padding default = 0, "VALID" anyways
                                if layer.pooling_param.pad == 0:
                                        new_node.attr["padding"].s = "VALID".encode("utf-8")
                                else:
                                        new_node.attr["padding"].s = "SAME".encode("utf-8")
                        except:
                                new_node.attr["padding"].s = "VALID".encode("utf-8")
                        try:
                                stride = layer.pooling_param.stride
                        except:
                                stride = 1
                        stride_list = [1, stride, stride, 1]
                        new_node.attr["strides"].list.CopyFrom(attr_value_pb2.AttrValue.ListValue(i=stride_list))
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]]) 

                        # if layer.name == "pool5/7x7_s1":
                        #         import code
                        #         code.interact(local=locals())
                        output_graph_def.node.extend([new_node])

                elif layer.type == "PriorBox":
                        # Follows definition of PriorBox class at https://github.com/intel/caffe/blob/master/src/caffe/layers/prior_box_layer.cpp
                        # Generate main node, we use a specific configuration of Reshape
                        new_node = node_def_pb2.NodeDef()
                        new_node.op = "Reshape"
                        new_node.name = layer.name
                        new_node.attr["T"].type = 1 # DT_INT32
                        new_node.attr["Tshape"].type = 3 # DT_INT32    
                        if len(layer.bottom) > 0:
                                new_node.input.extend([layer.bottom[0]])

                        # Get bottom's output shape
                        input_as_op = tf.import_graph_def(output_graph_def, return_elements=[layer.bottom[0]], name="")[0]
                        bottom_shape = input_as_op.outputs[0].shape.as_list()       

                        # Compute num_priors
                        min_size = len(layer.prior_box_param.min_size)
                        max_size = len(layer.prior_box_param.max_size)
                        aspect_ratio_size = len(layer.prior_box_param.aspect_ratio)
                        num_priors = min_size * aspect_ratio_size + max_size

                        # General case
                        num_dims = 3
                        out_dim = np.prod(bottom_shape[1:3])*num_priors*4

                        # 1 set of priors shared across all images in a batch
                        # 2 channels. 1st stores mean of each prior coordinate, second stores variance of each prior coordinate              
                        #TODO Figure out how to set out_shape[2] = out_dim as a valid reshape. Pad?
                        out_shape = [1, 2, -1]

                        # Generate shape node
                        shape_node = node_def_pb2.NodeDef()
                        shape_node.op = "Const"
                        shape_node.name = new_node.name + "/shape"
                        shape_node.attr["dtype"].type = 3 # DT_FLOAT32
                        shape_tuple = tuple(out_shape)
                        pack_format = '<'+'l'*num_dims
                        shape_packed = struct.pack(pack_format, *shape_tuple)
                        shape_node.attr["value"].tensor.tensor_shape.dim.add(size=num_dims)
                        shape_node.attr["value"].tensor.dtype = 3 # DT_INT32
                        shape_node.attr["value"].tensor.tensor_content = shape_packed
                        new_node.input.extend([shape_node.name])

                        output_graph_def.node.extend([new_node])
                        output_graph_def.node.extend([shape_node]) 

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
                        if len(unsorted_caffe_shape) == 4:
                                caffe_shape = [unsorted_caffe_shape[0],
                                                        unsorted_caffe_shape[2],
                                                        unsorted_caffe_shape[3],
                                                        unsorted_caffe_shape[1]]
                        else:
                                caffe_shape = unsorted_caffe_shape
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

## -------------------------------- MAIN ---------------------------------- ##
parser = argparse.ArgumentParser(description='Generates a TensorFlow model from a Caffe prototxt.')
parser.add_argument('-m', '--model', required=True, help='Target Caffe prototxt. e.g. deploy.prototxt')
parser.add_argument('-o', '--output', default='converted_caffe_model.pb', help='Name of output TensorFlow model. Default is converted_caffe_model.pb.')
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
with open(args.output, "wb") as f:
        f.write(output_graph_def.SerializeToString())
if len(unsupported_caffe_types) == 0:
        print('All caffe layer types in this prototxt are supported')
else:
        print('Unsupported Caffe ops: ', unsupported_caffe_types)
f.close()
