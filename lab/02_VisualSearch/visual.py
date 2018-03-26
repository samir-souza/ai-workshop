
import json
import logging
import os
import time
import numpy as np
import mxnet as mx

from collections import namedtuple

def train(current_host, hosts, num_cpus, num_gpus, channel_input_dirs, model_dir, hyperparameters, **kwargs):
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    mx.random.seed(127)
    
    path='http://data.mxnet.io/models/imagenet-11k/'
    [mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json', dirname='/tmp'),
     mx.test_utils.download(path+'resnet-152/resnet-152-0000.params', dirname='/tmp'),
     mx.test_utils.download(path+'synset.txt', dirname='/tmp')]
    
    sym, arg_params, aux_params = mx.model.load_checkpoint('/tmp/resnet-152', 0)
    
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
        label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    
    all_layers = sym.get_internals()

    hash_output = all_layers['flatten0_output']
    hash_output = mx.symbol.LogisticRegressionOutput(data=hash_output, name='sig')

    net = mx.symbol.Group([hash_output, all_layers["softmax_output"]])

    image_search_mod = mx.mod.Module(symbol=net, context=ctx, label_names=[ 'sig_label', 'softmax_label'])
    image_search_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
        label_shapes=image_search_mod._label_shapes)
    image_search_mod.set_params(arg_params, aux_params, allow_missing=False)
    
    return image_search_mod

def save(net, model_dir):
    net.save_checkpoint('%s/model' % model_dir, 0)
    
    shapes = open ( '%s/model-shapes.json' % model_dir, "w")
    json.dump([{"shape": net.data_shapes[0][1], "name": "data"}], shapes)
    shapes.flush()
    shapes.close()

def get_test_data(data_dir, batch_size, data_shape):
    return None

def get_train_data(data_dir, batch_size, data_shape):
    return None

def test(ctx, net, test_data):
    return None

def model_fn(model_dir):
    """
    Load the model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a network)
    """
    net, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, 'model'), 0)

    image_search_mod = mx.mod.Module(symbol=net, context=mx.cpu(), label_names=[ 'sig_label', 'softmax_label'])
    image_search_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
        label_shapes=image_search_mod._label_shapes)
    image_search_mod.set_params(arg_params, aux_params, allow_missing=False)

    return image_search_mod

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the model. Called once per request.

    :param net: The model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    resp = []

    try:
        Batch = namedtuple('Batch', ['data'])
        
        parsed = json.loads(data)
        img = mx.nd.array([parsed])
        
        net.forward(Batch([img]))
        prob_hash = net.get_outputs()[0][0].asnumpy()

        prob_cat = net.get_outputs()[1][0].asnumpy()
        prob_cat = np.squeeze(prob_cat)
        index_cat = np.argsort(prob_cat)[::-1]
        categories = []
        for i in index_cat[0:10]:
            categories.append( [int(i), float(prob_cat[i]) ] )

        hash_ =  "".join( map(str, np.where(prob_hash >= 0.75, 1, 0) ) )
    except Exception as e:
        logging.error(e)

    return json.dumps({"categories": categories, "hash": hash_ }), output_content_type

