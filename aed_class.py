from __future__ import print_function

import sys
import os
import time
import getopt
import argparse
import utils
import random

import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.metrics import f1_score
from lasagne import layers
import layers as cl
from sklearn import preprocessing as pp
import models

# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu3')

def parse_params(pdict):
    def change_param(key):
        if (key in pdict):
            PD[key] = type(PD[key])(pdict[key])
            print("Set {} to {}".format(key, PD[key]))
    for p in PD:
        change_param(p)

def setCNNParameters(param_file):
    if os.path.isfile(param_file):
        param_dict = {}
        with open(param_file) as pf:
            for line in pf:
                try: _name, _val = [i.strip() for i in line.split('=')]
                except: continue
                param_dict[_name] = _val
        parse_params(param_dict)
    else: 
        print("{} does not exist.".format(param_file))




# ##################### Build the neural network model #######################
def Han_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, None, 128),
                                        input_var=input_var)

    for i in [64, 128]:
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=i, 
                filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=i, 
                filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, 
                pool_size=(3, 3))
        network = lasagne.layers.dropout(network, p=0.3)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, 
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full')
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, 
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full')
    network = lasagne.layers.GlobalPoolLayer(network, pool_function=T.max)
    
    network = lasagne.layers.DenseLayer(
            network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.2),
            num_units=PD[OUTPUT_NUMBER],
            nonlinearity=lasagne.nonlinearities.softmax)
    return network



def JY_cnn(input_var_list, gaussian, delta):
    
    n_sources = len(input_var_list)
    n_early_conv = 2
    early_pool_size = 4

    d = early_pool_size**n_early_conv
    num_feat_type = 2 if delta else 1

    network_options = {
        'early_conv_dict_list': [
            {'conv_filter_list': [(60, 5) for ii in range(n_early_conv)],
             'pool_filter_list': [early_pool_size
                                  for ii in range(n_early_conv)],
             'pool_stride_list': [None for ii in range(n_early_conv)]}
            for ii in range(n_sources)
        ],
        'late_conv_dict': {
            'conv_filter_list': [(128, 1), (128, 1)],
            'pool_filter_list': [None, None],
            'pool_stride_list': [None, None]
        },
        'dense_filter_size': 1,
        'scan_dict': {
            'scan_filter_list': [32],
            'scan_std_list': [32/d],
            'scan_stride_list': [1],
        },
        'final_pool_function': T.max,  # T.max
        'input_size_list': [128 for nn in range(n_sources)],
        'output_size': 10,
        'p_dropout': 0.5,
        'num_feat_type': num_feat_type
    }

    print('n_early_conv: {}'.format(n_early_conv))
    print('early_pool_size: {}'.format(early_pool_size))
    print(network_options)

    if gaussian:
        return models.fcn_gaussian_multiscale(input_var_list, **network_options)
    else:
        return models.fcn_multiscale(input_var_list, **network_options)



### These functions are for iterating of separated data
def get_data_num(data_type, stdfeat):
    num = 0
    for root, dirs, files in os.walk('var/stdfeat/'+stdfeat):
        for fl in files:
            if data_type in fl:
                num += 1
    return num

def iter_data(data_type, num, stdfeat):
    for i in range(0, num):
        fn = data_type + '_' + str(i) + '.npy'
        fp = os.path.join('var/stdfeat', stdfeat, fn)
        yield np.load(fp)
### 

def get_test_batches(stdfeat):
    te_num = get_data_num('test', stdfeat)
    test_batches = []
    for n in range(te_num):
        fn = 'test_' + str(n) + '.npy'
        fp = os.path.join('var/stdfeat', stdfeat, fn)
        if len(test_batches) == 0:
            test_batches = np.load(fp)
        else:
            test_batches = np.append(test_batches, np.load(fp), axis=0)
    return test_batches


def main(stdfeat, num_epochs=300, param_file=None, 
         model_file=None, testing_type='8k', show_result=False, 
         frame_level='', continue_train=False, delta=False,
         gaussian=True, scales=3):
    # Load the dataset
    print("Loading data...")
    test_only = ( model_file and os.path.isfile(model_file) )
    # train_batches, val_batches, test_batches = load_dataset(training_type, testing_type, 
    #                                             fold, augment, delta, test_only, model_file)
    # print('train batches: {}'.format(len(train_batches)))
    # print('val batches: {}'.format(len(val_batches)))
    # print('test batches: {}'.format(len(test_batches)))

    # for bats, idx in my_iterator(train_batches, 100):
    #     np.save('var/stdfeat/train_'+str(idx), bats)
    # for bats, idx in my_iterator(val_batches, 100):
    #     np.save('var/stdfeat/val_'+str(idx), bats)
    # for bats, idx in my_iterator(test_batches, 100):
    #     np.save('var/stdfeat/test_'+str(idx), bats)

    # return
    test_batches = get_test_batches(stdfeat)
    

    print("Building model and compiling functions...")
    # Prepare Theano variables for inputs and targets
    n_sources = scales
    input_var_list = [T.tensor4('inputs{}'.format(i))
                      for i in range(n_sources)]
    target_var = T.imatrix('targets')

    # Create network model
    network = JY_cnn(input_var_list, gaussian, delta)


    print('defining loss function')
    prediction = lasagne.layers.get_output(network)
    prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    print('defining update')
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.005, momentum=0.9)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=0.008)
    

    print('defining testing method')
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.clip(test_prediction, 1e-7, 1.0 - 1e-7)

    #frame prediction
    layer_list = lasagne.layers.get_all_layers(network)
    gauss_layer = layer_list[-3]
    pre_gauss_layer = layer_list[-4] if gaussian else layer_list[-3]
    gauss_pred = lasagne.layers.get_output(gauss_layer, deterministic=True)
    pre_gauss_pred = lasagne.layers.get_output(pre_gauss_layer, deterministic=True)


    test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_pred_result = T.argmax(test_prediction, axis=1)
    target_result = T.argmax(target_var, axis=1)
    test_acc = T.mean(T.eq(test_pred_result, target_result),
                      dtype=theano.config.floatX)

    print('compiling training function')
    train_fn = theano.function(input_var_list + [target_var], 
                [loss, prediction, gauss_pred, pre_gauss_pred], updates=updates)

    print('compiling validation and testing function')
    val_fn = theano.function(input_var_list + [target_var], 
                [test_loss, test_acc, test_pred_result, test_prediction, gauss_pred, pre_gauss_pred])

    if not test_only or continue_train:
        if continue_train and os.path.isfile(model_file):
            print('Continue training {}'.format(model_file))
            val = np.load(model_file)
            lasagne.layers.set_all_param_values(network, val)

        print("Starting training...")
        lowest_err = 100000.0
        highest_acc = 0.0
        temp_err_filename =  '.temp_err_model_{}.npy'.format(int(time.time()*100000)) if model_file == None \
                    else '.temp_err_{}'.format(os.path.basename(model_file))
        temp_acc_filename = '.temp_acc_model_{}.npy'.format(int(time.time()*100000)) if model_file == None \
                    else '.temp_acc_{}'.format(os.path.basename(model_file))
        for epoch in range(num_epochs):
            # np.random.shuffle(train_batches)
            train_err = 0
            start_time = time.time()
            
            tr_num = get_data_num('train', stdfeat)
            no_tr = 0
            for train_batches in iter_data('train', tr_num, stdfeat):
                np.random.shuffle(train_batches)
                for batch in train_batches:
                    inputs, targets, names = zip(*batch)
                    inputs = [np.array(zip(*inputs)[x]) for x in range(n_sources)]
                    if inputs[0].shape[2] > 200000:
                        continue
                    targets = np.array(targets, dtype=np.int32)
                    inputs.append(targets)
                    err, pred, g_pred, pg_pred = train_fn(*inputs)
                    train_err += err
                    no_tr += 1
            print('Total number of training data: {}'.format(no_tr))

            val_err = 0
            val_acc = 0
            out_list, tar_list = [], []
            result_map = np.zeros((10,10), dtype=np.int32)
            va_num = get_data_num('val', stdfeat)
            no_va = 0
            v_ = [[],[],[],[],[]]
            for val_batches in iter_data('val', va_num, stdfeat):
                for batch in val_batches:
                    inputs, targets, names = zip(*batch)
                    inputs = [np.array(zip(*inputs)[x]) for x in range(n_sources)]
                    if inputs[0].shape[2] > 200000:
                        continue
                    targets = np.array(targets, dtype=np.int32)
                    inputs.append(targets)
                    err, acc, pred, pred_prob, g_pred, pg_pred  = val_fn(*inputs)
                    # print('out:\n {}'.format(out))
                    # print('targets:\n {}'.format(targets))
                    val_err += err
                    val_acc += acc
                    no_va += 1
                    # print((err, acc, pred_prob))
                    for i, j in zip(targets, pred):
                        result_map[i.argmax()][j] += 1
            print('Total number of validation data: {}'.format(no_va))

            # Print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / no_tr))
            print("  validation loss:\t\t{:.6f}".format(val_err / no_va))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / no_va * 100))
            print("Result map: (x: prediction, y: target)")
            print(result_map)

            # Save the best model
            if (val_err / no_va) <= lowest_err:
                lowest_err = val_err / no_va
                final_err_param = lasagne.layers.get_all_param_values(network)
                np.save(temp_err_filename, final_err_param)
            if (val_acc / no_va) >= highest_acc:
                highest_acc = val_acc / no_va
                final_acc_param = lasagne.layers.get_all_param_values(network)
                np.save(temp_acc_filename, final_acc_param)

            if epoch%25 == 24:
                run_test(test_batches, val_fn, n_sources, testing_type, frame_level, print_prob=True)

        if model_file:
            np.save('model/final/' + model_file, lasagne.layers.get_all_param_values(network))
        # return to the best weights
        val = np.load(temp_acc_filename)
        lasagne.layers.set_all_param_values(network, val)
        if model_file:
            os.rename(temp_acc_filename, 'model/acc/' + model_file)
            os.rename(temp_err_filename, 'model/err/' + model_file)
    else:
        print('Load weights from file {}'.format(model_file))
        val = np.load(model_file)
        lasagne.layers.set_all_param_values(network, val)

    run_test(test_batches, val_fn, n_sources, testing_type, frame_level)


def run_test(test_batches, test_fn, n_sources, testing_type, frame_level='', print_prob=False):
    result_map = np.zeros((10,10), dtype=np.int32)
    test_err = 0
    test_acc = 0
    res_list = []
    for batch in test_batches:
        inputs, targets, names = zip(*batch)
        inputs = [np.array(zip(*inputs)[x]) for x in range(n_sources)]
        targets = np.array(targets, dtype=np.int32)
        inputs.append(targets)
        err, acc, pred, pred_prob, g_pred, pg_pred = test_fn(*inputs)
        test_err += err
        test_acc += acc
        for i, j in zip(targets, pred):
            result_map[i.argmax()][j] += 1
        if print_prob and random.randint(0, 19) == 0:
            print(pred_prob)
            print(targets)
        if frame_level and frame_level != '':
            for n, t, p, gp, pgp in zip(names, targets, pred_prob, g_pred, pg_pred):
                # up-sampling
                gp = np.repeat(gp.reshape(10, -1), 16, axis=-1)
                pgp = np.repeat(pgp.reshape(10, -1), 16, axis=-1)

                if testing_type == 'us':
                    annotation = utils.load_us_annotation(n, np.argmax(t), gp.shape[-1])
                elif testing_type == 'tw':
                    annotation = utils.load_us_annotation(n, np.argmax(t), gp.shape[-1])
                else:
                    annotation = utils.load_us8k_annotation(n, np.argmax(t), gp.shape[-1])

                res = { 'name': str(n) ,'target': t, 'pred': p,
                        'g_pred': gp, 'pg_pred': pgp, 'annotation': annotation}
                res_list.append(res)
    if frame_level and frame_level != '':
        res_dir = os.path.join('var/frame_level_result', frame_level)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        np.save(os.path.join(res_dir, 'final.npy'), res_list)
                
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / len(test_batches)))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / len(test_batches) * 100))
    print("Result map: (x: prediction, y: target)")
    print(result_map)


def parser():
    p = argparse.ArgumentParser(description="Audio Event Detection using Fully Convolutional Network")
    p.add_argument('-S', '--stdfeat', type=str, metavar='standardized_feature',
                    help='Standardized feature')
    # p.add_argument('-l', '--training', type=str, metavar='training_type',
    #                 help='Training type')
    
    p.add_argument('-t', '--testing', type=str, metavar='testing_type',
                    help='Testing type')
    
    p.add_argument('-r', '--rd', action='store_true', default=False,
                    help='Show result data')
    
    p.add_argument('-e', '--epoch', type=int, 
                    help='Number of epochs')
    
    p.add_argument('-s', '--scales', type=int, default=3,
                    help='Number of scales (1, 2, or 3)')
    
    p.add_argument('-m', '--model', type=str, 
                    help='A pretrained model file')
    
    # p.add_argument('-s', '--std', action='store_true', default=True,
    #                 help='Standardize data')
    
    p.add_argument('-f', '--frame', type=str,
                    help='Output frame level prediction')
    
    p.add_argument('-c', '--continue_train', action='store_true', default=False,
                    help='Continue to train a model')

    p.add_argument('-F', '--fold', type=str,
                    help='Test folder')

    p.add_argument('-a', '--augment', action='store_true', default=False,
                    help='Train with data augmentation')

    p.add_argument('-g', '--gaussian', action='store_true', default=False,
                    help='Add gaussian layer to model')

    p.add_argument('-d', '--delta', action='store_true', default=False,
                    help='account delta as feature')


    args = p.parse_args()
    return args


def usage():
    print("Audio Event Detection using Fully Convolutional Network")


if __name__ == '__main__':
    kwargs = {}
    args = parser()
    kwargs['stdfeat'] = args.stdfeat
    kwargs['show_result'] = args.rd
    kwargs['model_file'] = args.model
    kwargs['num_epochs'] = args.epoch
    # kwargs['training_type'] = args.training
    kwargs['testing_type'] = args.testing
    kwargs['frame_level'] = args.frame
    kwargs['continue_train'] = args.continue_train
    # kwargs['fold'] = args.fold
    # kwargs['augment'] = args.augment
    kwargs['gaussian'] = args.gaussian
    # kwargs['scales'] = args.scales
    kwargs['delta'] = args.delta

    if args.scales not in [1, 2, 3]:
        print('Number of scales should be 1, 2, or 3. Default to 3.')
        

    print('training data: {}'.format(kwargs['training_type']))
    print('testing data: {}'.format(kwargs['testing_type']))
    print('epochs: {}'.format(kwargs['num_epochs']))
    print('stdfeat: {}'.format(kwargs['stdfeat']))
    print('model file: {}'.format(kwargs['model_file']))

    # if len(sys.argv[1:]) > 0:
        
    #     try:
    #         opts, args = getopt.getopt(sys.argv[1:], "hsp:e:m:n:l:t:",
    #                              ["help", "show_result_data", "parameter", "epoch", 
    #                               "model", "network", "training", "testing"])
    #     except getopt.GetoptError as err:
    #         print(str(err))
    #         usage()
    #     for o, a in opts:
    #         if o in ('-h', '--help'):
    #             usage()
    #             sys.exit()
    #         elif o in ('-s', '--show_result_data'):
    #             kwargs['show_result'] = True
    #         elif o in ('-p', '--parameter'):
    #             kwargs['param_file'] = a
    #         elif o in ('-e', '--epoch'):
    #             kwargs['num_epochs'] = int(a)
    #         elif o in ('-m', '--model'):
    #             kwargs['model_file'] = a
    #         # elif o in ('-n', '--network'):
    #         #     kwargs['network_model'] = a
    #         elif o in ('-l', '--training'):
    #             kwargs['training_file'] = a
    #         elif o in ('-t', '--testing'):
    #             kwargs['testing_file'] = a

    main(**kwargs)    
 
