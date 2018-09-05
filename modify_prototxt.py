#!/usr/bin/env python3

import sys
import argparse
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from collections import defaultdict

blob_layer_dict = defaultdict(str)

def get_parent(layer_bottom):
    if blob_layer_dict[layer_bottom]:
        return blob_layer_dict[layer_bottom]
    else:
        return layer_bottom
 
def add_to_dict(layer_top, top_value):
    blob_layer_dict[layer_top] = top_value


def modify_prototxt(args):

    prototxt = args.prototxt
    prototxtfile = open(prototxt,'r')

    net = caffe.Net(prototxt, caffe.TEST)
    
    net_par = caffe_pb2.NetParameter()
    text_format.Merge(prototxtfile.read(),net_par)

    mod_net = caffe_pb2.NetParameter()
    mod_net.name = net_par.name

    for l in net_par.layer:
        
        ltemp = mod_net.layer.add()
        ltemp.CopyFrom(l)

        #Needs to be done only for one top case
        #There are cases where Input layer have top with a different name
        if len(ltemp.top) == 1 and ltemp.type!='Input':
            ltemp.top[0] = l.name


        if len(ltemp.bottom) > 1:
            print('Inside multiple bottoms')
            print(ltemp.bottom)
            for i in range(len(ltemp.bottom)):
                ltemp.bottom[i] = get_parent(l.bottom[i])

            print('After modification', ltemp.bottom)

        elif len(ltemp.bottom) == 1:
            ltemp.bottom[0] = get_parent(l.bottom[0])
        else:
            print('Layer:%s has no bottom' % (ltemp.name))

        #Not to do for input layer since input layer can have top with different name (exp:FCN8s)
        if len(l.top) == 1 and l.type!='Input':
            add_to_dict(l.top[0], l.name)

    with open(net_par.name + '_tmp.prototxt','w') as f:
        f.write(text_format.MessageToString(mod_net))

def main(args):

    parser = argparse.ArgumentParser(description='Modify prototxt to remove in-place layers')
    parser.add_argument('-p', '--prototxt', type=str, help='Prototxt file')
    args = parser.parse_args(args)
    blob_layer_dict.clear()

    modify_prototxt(args)

if __name__=='__main__':
    main(sys.argv[1:])
