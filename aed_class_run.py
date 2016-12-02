import theano
import theano.tensor as T
import lasagne
import collections
import time
import sys
import os
import numpy as np
import random
import utils
# import theano.sandbox.cuda

global scales, train_fn, val_fn
scales = 3

def JY_cnn(input_var_list, gaussian, delta):
    import models
    import functions
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


def init_process(model, gaussian, delta, fn_type):
    print("Building model and compiling functions...")
    # Prepare Theano variables for inputs and targets
    import theano.tensor as T
    input_var_list = [T.tensor4('inputs{}'.format(i))
                      for i in range(scales)]
    target_var = T.imatrix('targets')

    # Create network model
    if model == 'jy':
        print('Building JY CNN...')
        network = JY_cnn(input_var_list, gaussian, delta)
        learning_rate = 0.006
    # elif model == 'fcrnn':
    #     print('Building FCRNN...')
    #     network = FCRNN(input_var_list, delta)
    #     learning_rate = 0.0005

    print('defining loss function')
    prediction = lasagne.layers.get_output(network)
    prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()

    print('defining update')
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)
    # updates = lasagne.updates.adagrad(loss, params, learning_rate=learning_rate)
    

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

    if fn_type == 'train':
        print('compiling training function')
        func = theano.function(input_var_list + [target_var], 
                    [loss, prediction, gauss_pred, pre_gauss_pred], updates=updates)
    elif fn_type == 'val' or fn_type == 'test':
        print('compiling validation and testing function')
        func = theano.function(input_var_list + [target_var], 
                    [test_loss, test_acc, test_pred_result, test_prediction, gauss_pred, pre_gauss_pred])

    return func, network

def get_batch_length(bt):
    return bt[0][0][0].shape[1]

def proc_run(lock, idx, fp_list, shared_param, dt_type):
    def process_batches(batch_queue, network, exec_func, func):
        print('[Queue {}] Start {} {} batches.\t{}'.format(idx, dt_type, len(batch_queue), utils.print_time()))
        lg = get_batch_length(batch_queue[0])
        st = time.time()
        n_elem = 0
        n = shared_param.finished_num
        if hasattr(shared_param, 'model'):
            lasagne.layers.set_all_param_values(network, shared_param.model)
        for bt in batch_queue:
            if type(bt) != type('end'):
                exec_func(bt, func, shared_param)
                n_elem += 1
            else:
                n += 1
                print('[Queue {}] Finished {}: {}*{}.\t{:.3f} secs.'.format(idx, n, lg, n_elem, time.time()-st))
                n_elem = 0
                sys.stdout.flush()
        batch_queue.clear()
        shared_param.finished_num = n
        shared_param.model = lasagne.layers.get_all_param_values(network)
        # ng = gc.collect()
        # del gc.garbage[:]
        

    model = shared_param.m_type
    gaussian = shared_param.gaussian
    delta = shared_param.delta
    print('[Queue {}] Initializing process.'.format(idx))
    func, network = init_process(model, gaussian, delta, dt_type)
    if dt_type == 'train':
        exec_func = proc_train_a_batch
    elif dt_type == 'val':
        exec_func = proc_val_a_batch
    batch_queue = collections.deque()
    while len(fp_list) > 0:
        if len(batch_queue) > 0:
            if lock.acquire(False):
                process_batches(batch_queue, network, exec_func, func)
                lock.release()
        elif len(batch_queue) > 200:
            sleep(5)
            continue
        fp = fp_list.pop()
        proc_loadfile(batch_queue, idx, fp)
    if len(batch_queue) > 0:
        lock.acquire()
        process_batches(batch_queue, network, exec_func, func)
        lock.release()

def proc_train_a_batch(batch, train_fn, shared_param):
    inputs, targets, names = batch
    if inputs[0].shape[2] > 200000:
        return
    inputs.append(targets)
    try:
        err, pred, g_pred, pg_pred = train_fn(*inputs)
        if err is not None:
            shared_param.train_err += err*len(pred)
            shared_param.no_tr += len(pred)
        inputs.pop()
    except  Exception as error: 
        print(error)

def proc_val_a_batch(batch, val_fn, shared_param):
    inputs, targets, names = batch
    if inputs[0].shape[2] > 200000:
        return
    inputs.append(targets)
    try:
        r_map = shared_param.result_map
        err, acc, pred, pred_prob, g_pred, pg_pred = val_fn(*inputs)
        if err is not None:
            shared_param.val_err += err*len(pred)
            shared_param.val_acc += acc*len(pred)
            shared_param.no_va += len(pred)
            # print((err, acc, pred_prob))
            for i, j in zip(targets, pred):
                r_map[i.argmax()][j] += 1
        inputs.pop()
        shared_param.result_map = r_map
    except  Exception as error: 
        print(error)

def proc_loadfile(batch_queue, i, fp):
    # batch_queue, lock, i, fp = args
    # name = multiprocessing.current_process().name
    while len(batch_queue) > 500:
        time.sleep(5)
        # print('{} sleep for 60 seconds.'.format(name))
    fn = os.path.basename(fp)
    # print('Load {} to {}.'.format(fn, i))
    st = time.time()
    bts = load_batch_file(fp)

    # Save to cache if the file is large
    bl = get_batch_length(bts[0])
    # if bl * len(bts) >= 300000:
    #     print('[Queue {}] Cache {}'.format(i, fn))
    #     dt_type, dt_n = fn.replace('.npy', '').split('_')
    #     cache_lock.acquire()
    #     cache_data_dict[dt_type][dt_n] = bts
    #     cache_lock.release()

    np.random.shuffle(bts)
    # lock.acquire()
    # print('{} got the lock {}.'.format(fn, i))
    for bt in bts:
        batch_queue.append(bt)
    batch_queue.append('end')
    # print('Pool finished {}. {}'.format(fn, utils.print_time()))
    # print('Queue {} size: {}.'.format(i, len(batch_queue)))
    print('[Queue {}] {}: [{}*{}].\t{:.4f} secs.'.format(i, fn, bl, len(bts), time.time()-st))
    del bts
    # print('{} releasing the lock {}.'.format(fn, i))
    # lock.release()

def load_batch_file(fp):
    bts = np.load(fp)
    batches = [(bt['inputs'], bt['targets'], bt['names']) for bt in bts]
    return batches

def get_data_num(data_type, stdfeat):
    num = 0
    for root, dirs, files in os.walk('var/stdfeat/'+stdfeat):
        for fl in files:
            if data_type in fl:
                num += 1
    return num

def get_test_batches(stdfeat):
    te_num = get_data_num('test', stdfeat)
    test_batches = []
    for n in range(te_num):
        fn = 'test_' + str(n) + '.npy'
        fp = os.path.join('var/stdfeat', stdfeat, fn)
        test_batches += load_batch_file(fp)
    return test_batches


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
        print('test_a_batch returning None!')
        print(error)
        sys.exit()
        return None, None, None, None, None, None, None, None

def proc_test(shared_param, stdfeat, testing_type, frame_level='', print_prob=False):
    # import theano
    # import theano.tensor as T
    # import lasagne
    test_batches = get_test_batches(stdfeat)
    
    # build networks and functions
    model = shared_param.m_type
    gaussian = shared_param.gaussian
    val_fn, network = init_process(model, gaussian, delta, 'test')

    lasagne.layers.set_all_param_values(network, shared_param.model)
    run_test(test_batches, val_fn, testing_type, frame_level, print_prob)



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