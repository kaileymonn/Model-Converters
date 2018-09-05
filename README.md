# Converts Caffe/Onnx models to TensorFlow graphs #
Built for object detection models, other workloads may not be ideal. Mainly for visualizing non-TensorFlow models on TensorBoard, as well as other TensorFlow-specific tools.

## Introduction ##
This tool takes a Caffe prototxt/Onnx model and generates a TensorFlow GraphDef which is serialized and written to a protobuf (converted_model.pb by default).  Not configured for inference use.

### Quick start ###
Common usage involves running the following commands:   
* $ python3 caffe2tf.py -m path/to/deploy.prototxt or
* $ python3 onnx2tf.py -m path/to/model.onnx
  - converted_model.pb (default name) is generated in the same directory
  - Caffe models only require a .prototxt file. Caffemodel files are not required.
  - Onnx models only require a .onnx file.

### Arguments ###
* -m : This is a required argument reflecting the path to your Caffe prototxt/Onnx model file   
* -o : This is an optional argument to set the output TensorFlow protobuf's name

### Files ###
- caffe2tf.py
- onnx2tf.py
