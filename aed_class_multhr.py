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
import signal
import threading
from threading import Thread
import multiprocessing
from multiprocessing import Process, Queue, Manager, Lock, Pool
import collections
import gc

import functions 

# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu3')

global scales
global all_data_queues
global cache_data
all_data_queues = {'train':[], 'val':[]}

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
        'final_pool_function': functions.gated_mean,  # T.max
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


def FCRNN(input_var_list, delta):
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
        'final_pool_function': T.max,  # T.max
        'input_size_list': [128 for nn in range(n_sources)],
        'last_late_conv_size' : 128,
        'output_size': 10,
        'p_dropout': 0.5,
        'num_feat_type': num_feat_type,
        'num_lstm_unit': 64,
        'gradient_steps' : 10
    }

    print('n_early_conv: {}'.format(n_early_conv))
    print('early_pool_size: {}'.format(early_pool_size))
    print(network_options)

    return models.fcrnn(input_var_list, **network_options)



### These functions are for iterating of separated data
def get_data_num(data_type, stdfeat):
    num = 0
    for root, dirs, files in os.walk('var/stdfeat/'+stdfeat):
        for fl in files:
            if data_type in fl:
                num += 1
    return num

def get_fp_list(stdfeat, epochs, shuffle=False):
    def generate_fp_list(data_type, fp_list):
        indices = range(0, get_data_num(data_type, stdfeat))
        if shuffle: 
            random.shuffle(indices)
        for i in indices:
            fn = data_type + '_' + str(i) + '.npy'
            fp_list.append(os.path.join('var/stdfeat', stdfeat, fn))
    fp_list = []
    for e in range(epochs):
        generate_fp_list('train', fp_list)
        generate_fp_list('val', fp_list)
    print('fp_list length: {}'.format(len(fp_list)))
    return fp_list

def iter_batch(batch_queue, data_type, num):
    for i in range(num):
        while True:
            sleep_time = 0
            # print('queue size: {}'.format(len(batch_queue)))
            # print('queue size: {}'.format(batch_queue.qsize()))
            while len(batch_queue) == 0:
            # while batch_queue.empty():
                # tt = 10 if batch_queue.qsize()==0 else 1
                tt = 10
                time.sleep(tt)
                sleep_time += tt
                # print('sleep {} seconds.'.format(sleep_time))
            # if sleep_time > 0: 
                # print('sleep {} seconds.'.format(sleep_time))
            bt = batch_queue.pop(0)
            # bt = batch_queue.get(block=False)
            if type(bt) != type('end'):
                # print('get one batch!')
                yield bt
            else:
                print('iter end one fp! {}'.format(utils.print_time()))
                name = multiprocessing.current_process().name
                print('{} current queue size: {}'.format(name, len(batch_queue)))
                break
    # print('End iteration of {}'.format(data_type))
### 

def th_loadfile(batch_queue, fp_list):
    name = multiprocessing.current_process().name
    for fp in fp_list:
        while len(batch_queue) > 100:
        # while batch_queue.full():
            time.sleep(5)
            # print('{} sleep for 60 seconds.'.format(name))
        bts = np.load(fp)
        # print('{} originally has {} batches.'.format(fp, len(bts)))
        np.random.shuffle(bts)
        # print('{} has {} batches to put.'.format(fp, len(bts)))
        for bt in bts:
            batch_queue.append(bt)
            # batch_queue.put(bt, block=True)
        batch_queue.append('end')
        # batch_queue.put('end', block=True)
        print('{} finished {}. {}'.format(name, fp, utils.print_time()))
        print('{} current queue size: {}'.format(name, len(batch_queue)))

    print('{} ends.'.format(name))



def get_test_batches(stdfeat):
    te_num = get_data_num('test', stdfeat)
    test_batches = []
    for n in range(te_num):
        fn = 'test_' + str(n) + '.npy'
        fp = os.path.join('var/stdfeat', stdfeat, fn)
        test_batches += load_batch_file(fp)
    return test_batches
    # for bt in bts:
    #     test_batches.append(bt)


def init_process(model, gaussian, delta):
    print("Building model and compiling functions...")
    # Prepare Theano variables for inputs and targets
    input_var_list = [T.tensor4('inputs{}'.format(i))
                      for i in range(scales)]
    target_var = T.imatrix('targets')

    # Create network model
    if model == 'jy':
        print('Building JY CNN...')
        network = JY_cnn(input_var_list, gaussian, delta)
        learning_rate = 0.006
    elif model == 'fcrnn':
        print('Building FCRNN...')
        network = FCRNN(input_var_list, delta)
        learning_rate = 0.0005

    print('defining loss function')
    prediction = lasagne.layers.get_output(network)
    prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    print('defining update')
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.005, momentum=0.9)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=learning_rate)
    

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

    return train_fn, val_fn, network

def chop_batches(batches, bsize):
    new_bts = []
    for bt in batches:
        for i in range(0, len(bt)/bsize+1):
            nbt = bt[bsize*i:bsize*(i+1)]
            if len(nbt) > 0:
                new_bts.append(nbt)
    return np.asarray(new_bts)

### Pool

# def initPool(b, l):
#     global batch_queue
#     batch_queue = b
#     global lock
#     lock = l

def load_batch_file(fp):
    bts = np.load(fp)
    batches = [(bt['inputs'], bt['targets'], bt['names']) for bt in bts]
    return batches

def proc_loadfile(batch_queue, lock, idx, fp_list, dt_type):
    for fp in fp_list:
        pool_loadfile((batch_queue, lock, idx, fp, dt_type))

def pool_loadfile(args):
    batch_queue, lock, i, fp, dt_type = args
    # name = multiprocessing.current_process().name
    while len(batch_queue) > 120:
        time.sleep(5)
        # print('{} sleep for 60 seconds.'.format(name))
    fn = os.path.basename(fp)
    # print('Load {} to {}.'.format(fn, i))
    st = time.time()
    bts = load_batch_file(fp)
    # bts = chop_batches(bts, 5)
    bl = get_batch_length(bts[0])
    np.random.shuffle(bts)
    lock.acquire()
    # print('{} got the lock {}.'.format(fn, i))
    for bt in bts:
        batch_queue.append(bt)
        if cache_data:
            all_data_queues[dt_type].append(bt)
    batch_queue.append('end')
    # print('Pool finished {}. {}'.format(fn, utils.print_time()))
    # print('Queue {} size: {}.'.format(i, len(batch_queue)))
    print('[Queue {}] {}: [{}*{}].\t{:.4f} secs.'.format(i, fn, bl, len(bts), time.time()-st))
    del bts
    # print('{} releasing the lock {}.'.format(fn, i))
    lock.release()

# def reshape_batch(batch):
#     inputs, targets, names = zip(*batch)
#     tmp_in = zip(*inputs)
#     inputs = [np.array(tmp_in[x]) for x in range(scales)]
#     targets = np.array(targets, dtype=np.int32)
#     return (inputs, targets, names)

def generate_fp_list(data_type, stdfeat, shuffle=False):
    fp_list = []
    indices = range(0, get_data_num(data_type, stdfeat))
    if shuffle: 
        random.shuffle(indices)
    for i in indices:
        fn = data_type + '_' + str(i) + '.npy'
        fp_list.append(os.path.join('var/stdfeat', stdfeat, fn))
    return fp_list

def gen_pool_fp_list(data_type, stdfeat, qlts, shuffle=True):
    fp_list = generate_fp_list(data_type, stdfeat, shuffle)
    l = len(qlts)
    n_fp_list = []
    for idx, fp in enumerate(fp_list):
        queue, lock, i = qlts[idx%l]
        n_fp_list.append((queue, lock, i, fp))
    random.shuffle(n_fp_list)
    return n_fp_list

def gen_procs(qlts, fp_list, dt_type):
    proc_list = []
    n_cores = len(qlts)
    for i in range(n_cores):
        queue, lock, idx = qlts[i] 
        fpl = fp_list[i::n_cores]
        name = 'loadfile_proc_'+str(i)
        # proc = Process(target=proc_loadfile, args=(queue, lock, idx, fpl),
        #                 name=name)
        proc = Thread(target=proc_loadfile, args=(queue, lock, idx, fpl, dt_type),
                        name=name)
        proc.start()
        proc_list.append(proc)
    return proc_list

def get_batch_length(bt):
    return bt[0][0][0].shape[1]

def new_iter_batch(qlts, num):
    n = 0
    n_cores = len(qlts)
    sleep_t = 0
    while n < num:
        empty = 0
        for qlt in qlts:
            batch_queue, lock, i = qlt
            if len(batch_queue) == 0:
                empty += 1
                continue
            if lock.acquire(False):
                print('[Main th] Slept for {} secs.'.format(sleep_t))
                # print('Taking batches from batch_queue {}'.format(i))
                # bts = [bt for bt in batch_queue]
                # batch_queue.clear()
                # del batch_queue[:]
                bts = []
                while len(batch_queue) > 0:
                    bts.append(batch_queue.popleft())
                # print('[Main th] Got {} batches from Queue {}. {}'.format(len(bts), i, utils.print_time()))
                lock.release()

                lg = 0
                st = time.time()
                n_elem = 0
                for bt in bts:
                    if type(bt) != type('end'):
                        lg = get_batch_length(bt)
                        # ti = time.time()
                        # print('GO TRAIN {}!'.format(lg))
                        yield bt
                        # print('END TRAIN {}! {}'.format(lg, time.time()-ti))
                        n_elem += 1
                    else:
                        n += 1
                        print('[Main th] Finished {}: {}*{}. {}'.format(n, lg, n_elem, time.time()-st))
                        n_elem = 0
                        st = time.time()
                        sys.stdout.flush()
                del bts
                # ng = gc.collect()
                # print('{} garbages.'.format(ng))
                # del gc.garbage[:]
            else:
                empty += 1
        if empty == n_cores:
            # print('All empty or unavailable.')
            time.sleep(2)
            sleep_t += 2
    # print('End iter.')
    print('[Main th] Totally slept {} secs.'.format(sleep_t))



    # n = 0
    # while n < num:
    #     if len(batch_queue) == 0:
    #         time.sleep(10)
    #         continue
    #     bts = []
    #     lock.acquire()
    #     print('Taking batches...')
    #     while len(batch_queue) > 0:
    #         bts.append(batch_queue.pop(0))
    #     print('Got {} batches from batch_queue.'.format(len(bts)))
    #     lock.release()
    #     for bt in bts:
    #         if type(bt) != type('end'):
    #             yield bt
    #         else:
    #             print('Got one end!')
    #             n += 1
    # print('End iter.')



    # for i in range(num):
    #     print('iter of {}'.format(i))
    #     sleep_time = 0
    #     while len(batch_queue) == 0:
    #         tt = 10
    #         time.sleep(tt)
    #         sleep_time += tt
    #         print('sleeping {} seconds.'.format(sleep_time))
    #     bts = []
    #     lock.acquire()
    #     while True:
    #         bts.append(batch_queue.pop(0))
    #         # lock.release()
    #         if type(bt) != type('end'):
    #             yield bt
    #         else:
    #             print('iter end one fp! {}'.format(utils.print_time()))
    #             name = multiprocessing.current_process().name
    #             print('{} current queue size: {}'.format(name, len(batch_queue)))
    #             break

def train_a_batch(batch, train_fn):
    # ti = time.time()
    inputs, targets, names = batch
    if inputs[0].shape[2] > 200000:
        return None, None, None, None
    inputs.append(targets)
    try:
        # print('start running yo!', time.time()-ti)
        err, pred, g_pred, pg_pred = train_fn(*inputs)
        # print('end running yo!', time.time()-ti)
        inputs.pop()
        return err, pred, g_pred, pg_pred
    except: 
        inputs.pop()
        return None, None, None, None

def val_a_batch(batch, val_fn):
    inputs, targets, names = batch
    if inputs[0].shape[2] > 200000:
        return None, None, None, None, None, None, None
    inputs.append(targets)
    try:
        err, acc, pred, pred_prob, g_pred, pg_pred = val_fn(*inputs)
        inputs.pop()
        return err, acc, pred, pred_prob, g_pred, pg_pred, targets
    except:
        inputs.pop()
        return None, None, None, None, None, None, None

def test_a_batch(batch, test_fn):
    inputs, targets, names = batch
    # if inputs[0].shape[2] > 200000:
    #     return None, None, None, None, None, None, None
    inputs.append(targets)
    try:
        err, acc, pred, pred_prob, g_pred, pg_pred = test_fn(*inputs)
        inputs.pop()
        return err, acc, pred, pred_prob, g_pred, pg_pred, targets, names
    except  Exception as error:
        inputs.pop()
        print('test_a_batch returning None!')
        print(error)
        sys.exit()
        return None, None, None, None, None, None, None, None


def main(stdfeat, num_epochs=300, param_file=None, 
         model_file=None, testing_type='8k', model='jy',
         frame_level='', continue_train=False, delta=False,
         gaussian=True):
    # gc.disable()

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
    
    # build networks and functions
    train_fn, val_fn, network = init_process(model, gaussian, delta)
    

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
        
        n_cores = 5
        qlts = []
        for i in range(n_cores):
            # mgr = Manager()
            qlts.append((collections.deque(), threading.Lock(), i))
        # pool = Pool(processes=n_cores)
        # train_queue, val_queue = mgr.list(), mgr.list()
        # train_queue, val_queue = Queue(2000), Queue(1000)
        # tr_fp_list = get_fp_list(stdfeat, num_epochs, True)
        # tr_fp_list = get_fp_list('train', stdfeat, num_epochs, True)
        # va_fp_list = get_fp_list('val', stdfeat, num_epochs, True)
        # load_data_proc = Process(target=th_loadfile, args=(train_queue, tr_fp_list), 
        #                           name='tr_loadfile')
        # load_train_proc = Process(target=th_loadfile, args=(train_queue, tr_fp_list), 
        #                           name='tr_loadfile')
        # load_val_proc = Process(target=th_loadfile, args=(val_queue, va_fp_list), 
        #                         name='va_loadfile')
        # load_data_proc.start()
        # load_train_proc.start()
        # load_val_proc.start()
        tr_num = get_data_num('train', stdfeat)
        va_num = get_data_num('val', stdfeat)
        sys.stdout.flush()
        for epoch in range(num_epochs):
            print('epoch {}. {}'.format(epoch, utils.print_time()))
            start_time = time.time()
            # np.random.shuffle(train_batches)
            train_err, no_tr, val_err, val_acc, no_va = 0, 0, 0, 0, 0
            out_list, tar_list = [], []
            result_map = np.zeros((10,10), dtype=np.int32)

            if cache_data and epoch != 0:
                np.random.shuffle(all_data_queues['train'])
                for batch in all_data_queues['train']:
                    err, pred, g_pred, pg_pred = train_a_batch(batch, train_fn)
                    if err is not None:
                        train_err += err*len(pred)
                        no_tr += len(pred)
                # print('Total number of training data: {}'.format(no_tr))
                for batch in all_data_queues['val']:
                    err, acc, pred, pred_prob, g_pred, pg_pred, targets = val_a_batch(batch, val_fn)
                    if err is not None:
                        val_err += err*len(pred)
                        val_acc += acc*len(pred)
                        no_va += len(pred)
                        for i, j in zip(targets, pred):
                            result_map[i.argmax()][j] += 1
                # print('Total number of validation data: {}'.format(no_va))

            else:
                # pool.map_async(pool_loadfile, gen_pool_fp_list('train', stdfeat, qlts, True))
                proc_list = gen_procs(qlts, generate_fp_list('train', stdfeat, True), 'train')
                for batch in new_iter_batch(qlts, tr_num):
                # for batch in iter_batch(train_queue, 'train', tr_num):
                    err, pred, g_pred, pg_pred = train_a_batch(batch, train_fn)
                    if err is not None:
                        train_err += err*len(pred)
                        no_tr += len(pred)
                # print('Join all procs.')
                for proc in proc_list:
                    proc.join()
                print('Total number of training data: {}'.format(no_tr))

                # pool.map_async(pool_loadfile, gen_pool_fp_list('val', stdfeat, qlts, False))
                proc_list = gen_procs(qlts, generate_fp_list('val', stdfeat, True), 'val')
                for batch in new_iter_batch(qlts, va_num):
                # for batch in iter_batch(train_queue, 'val', va_num):
                # for batch in iter_batch(val_queue, 'val', va_num):
                    err, acc, pred, pred_prob, g_pred, pg_pred, targets = val_a_batch(batch, val_fn)
                    # print('out:\n {}'.format(out))
                    # print('targets:\n {}'.format(targets))
                    if err is not None:
                        val_err += err*len(pred)
                        val_acc += acc*len(pred)
                        no_va += len(pred)
                        # print((err, acc, pred_prob))
                        for i, j in zip(targets, pred):
                            result_map[i.argmax()][j] += 1
                # print('Join all procs.')
                for proc in proc_list:
                    proc.join()
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
                run_test(test_batches, val_fn, testing_type, frame_level, print_prob=True)

            sys.stdout.flush()

        # load_train_proc.join()
        # load_val_proc.join()
        # pool.close()
        # pool.join()

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

    run_test(test_batches, val_fn, testing_type, frame_level)
    # gc.enable()


def run_test(test_batches, test_fn, testing_type, frame_level='', print_prob=False):
    result_map = np.zeros((10,10), dtype=np.int32)
    test_err = 0
    test_acc = 0
    num_te = 0
    res_list = []
    for batch in test_batches:
        err, acc, pred, pred_prob, g_pred, pg_pred, targets, names = test_a_batch(batch, test_fn)
        if err is None:
            continue
        test_err += err*len(pred)
        test_acc += acc*len(pred)
        num_te += len(pred)
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
    print("  test loss:\t\t\t{:.6f}".format(test_err / num_te))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / num_te * 100))
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

    p.add_argument('-C', '--cache_data', action='store_true', default=False,
                    help='Cahce the data')

    group = p.add_mutually_exclusive_group()
    group.add_argument("-j", "--jycnn", action="store_true")
    group.add_argument("-r", "--fcrnn", action="store_true")


    args = p.parse_args()
    return args


def usage():
    print("Audio Event Detection using Fully Convolutional Network")


if __name__ == '__main__':
    kwargs = {}
    args = parser()
    kwargs['stdfeat'] = args.stdfeat
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
    cache_data = args.cache_data
    if args.jycnn:
        kwargs['model'] = 'jy'
    elif args.fcrnn:
        kwargs['model'] = 'fcrnn'
    else:
        print('Default select to jycnn.')
        kwargs['model'] = 'jy'

    if args.scales not in [1, 2, 3]:
        scales = 3
        print('Number of scales should be 1, 2, or 3. Default to 3.')
    else:
        scales = args.scales
        

    print('testing data: {}'.format(kwargs['testing_type']))
    print('epochs: {}'.format(kwargs['num_epochs']))
    print('stdfeat: {}'.format(kwargs['stdfeat']))
    print('model file: {}'.format(kwargs['model_file']))
    print('cache_data: {}'.format(cache_data))

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
 
