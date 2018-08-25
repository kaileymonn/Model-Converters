# Converts Caffe models to TensorFlow graphs #
Copyright (c) 2018-2019 Ambarella, Inc.   
Feel free to make improvements or provide feedback if you find any issues! :)

## Introduction ##
This tool takes a Caffe prototxt and generates a TensorFlow GraphDef which is serialized and written to a protobuf (converted_model.pb by default). Great if you just want to visualize a Caffe model on Tensorboard for example. Not configured for inference use.

### Quick start ###
Common usage involves running the following commands:   
* $ python3 convert_caffe_to_tf.py --model=/path/to/deploy.prototxt   
  - converted_model.pb is generated in the same directory

### Arguments ###
* --model : This is a required argument reflecting the path to your Caffe prototxt file   
* --output : This is an optional argument to set the output model's name

### Overview ###
- Total estimates are printed at the top of the spreadsheet under the 'Key Values' table. 
- Possible input nodes and output nodes are listed under the 'Feeds : row #' and 'Fetches : row #' tables respectively, mapped to their corresponding row numbers in the spreadsheet. 
- Values listed under the 'Independent Variables' table are configurable, as well as values in column S ('MACs/Cycle'). 
- More default variable settings can be configured under settings.py, depending on the user's needs. 
- Color coding in columns A and B denote the following: 
  - Orange cells labelled 'Unsupported' in column A correspond to ops that are yet to be supported by this tool. 
  - Red cells in column B correspond to ops that not supported by tfparser (not included). Users can update tfparser_supported_ops.py.

### Files ###
- convert_caffe_to_tf.py
