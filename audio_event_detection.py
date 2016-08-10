from __future__ import print_function

import sys
import os
import time
import getopt

import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.metrics import f1_score


# ###### Define Variables ######


NO_FILTER_1 = 'NO_FILTER_1'
FILTER_1_ROW = 'FILTER_1_ROW'
FILTER_1_COL = 'FILTER_1_COL'
POOL_1_ROW = 'POOL_1_ROW'
POOL_1_COL = 'POOL_1_COL'
NO_FILTER_2 = 'NO_FILTER_2'
FILTER_2_ROW = 'FILTER_2_ROW'
FILTER_2_COL = 'FILTER_2_COL'
POOL_2_ROW = 'POOL_2_ROW'
POOL_2_COL = 'POOL_2_COL'
DROP_RATE_1 = 'DROP_RATE_1'
NO_UNIT_DEN_1 = 'NO_UNIT_DEN_1'
DROP_RATE_2 = 'DROP_RATE_2'

OUTPUT_NUMBER = 'OUTPUT_NUMBER'
BATCH_SIZE = 'BATCH_SIZE'

PD = {
    NO_FILTER_1 : 32,
    FILTER_1_ROW : 8,
    FILTER_1_COL : 128,
    POOL_1_ROW : 2,
    POOL_1_COL : 1,
    NO_FILTER_2 : 64,
    FILTER_2_ROW : 8,
    FILTER_2_COL : 1,
    POOL_2_ROW : 2,
    POOL_2_COL : 1,
    DROP_RATE_1 : .2,
    NO_UNIT_DEN_1 : 400,
    DROP_RATE_2 : .2,
    OUTPUT_NUMBER : 10,
    BATCH_SIZE : 10,
}




# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset(training_dir='training_data/feature', 
                 testing_dir='testing_data/feature', 
                 test_only=False):
    def load_features(feat_dir):
        pair_list = []
        for root, dirs, files in os.walk(feat_dir):
            if files:
                for in_fn in files:
                    if in_fn.endswith('npy'):
                        print('Loading {}'.format(in_fn))
                        _pair = np.load(os.path.join(root, in_fn))
                        pair_list.append(_pair)
        
        np.random.shuffle(pair_list)


        feature_list, tag_list = [], []
        for pair in pair_list:
            print(np.asarray(pair[0], dtype=np.float32).shape)
            feature_list.append(np.asarray(pair[0], dtype=np.float32))
            tag_list.append(np.asarray(pair[1], dtype=np.int32))
        np.savez('tt.npz', feature_list)
        feature_list = np.array(feature_list, dtype=object)
        tag_list = np.array(tag_list)
        return feature_list, tag_list 

    def classify_data():


    X_train, y_train, X_val, y_val = np.array([]), np.array([]), np.array([]), np.array([])
    if not test_only:
        X_train, y_train = load_features(training_dir)
        p1 = len(X_train)*9/10
        X_train, X_val = X_train[:p1], X_train[p1:]
        y_train, y_val = y_train[:p1], y_train[p1:]
    else:
        print('Loading only testing data')

    X_test, y_test = load_features(testing_dir)


    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

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
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    # columns = n_mels (= 128)
    network = lasagne.layers.InputLayer(shape=(None, 1, 121, 128),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 8x8. 
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=PD[NO_FILTER_1], 
            filter_size=(PD[FILTER_1_ROW], PD[FILTER_1_COL]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
   
    # Max-pooling layer of 2 in row and 2 in column:
    network = lasagne.layers.MaxPool2DLayer(network, 
                                            pool_size=(PD[POOL_1_ROW], PD[POOL_1_COL]))

    # Another convolution with 32 8x8 kernels, and another 1x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=PD[NO_FILTER_2], 
            filter_size=(PD[FILTER_2_ROW], PD[FILTER_2_COL]),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, 
                                            pool_size=(PD[POOL_2_ROW], PD[POOL_2_COL]))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=PD[DROP_RATE_1]),
            num_units=PD[NO_UNIT_DEN_1],
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=PD[DROP_RATE_2]),
            num_units=PD[OUTPUT_NUMBER],
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network

def Han_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, None, 128),
                                        input_var=input_var)

    for i in [32, 64, 128]:
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=i, 
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                pad='full',
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=i, 
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                pad='full')
        network = lasagne.layers.MaxPool2DLayer(network, 
                pool_size=(2, 2))
        network = lasagne.layers.dropout(network, p=0.25)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, 
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full')
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, 
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='full')
    network = lasagne.layers.GlobalPoolLayer(network, pool_function=T.max)
    
    network = lasagne.layers.DenseLayer(
            network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.sigmoid)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.2),
            num_units=PD[OUTPUT_NUMBER],
            nonlinearity=lasagne.nonlinearities.sigmoid)
    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



# windowsize represents the number of frames in an window
def iter_windows(inputs, windowsize, shuffle=False):
    for start_idx in range(0, len(inputs) - windowsize + 1, windowsize):
        excerpt = slice(start_idx, start_idx + windowsize)
        yield inputs[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=10, network_model='han_cnn', param_file=None, 
         model_file=None, training_dir='training_data/feature',
         testing_dir='testing_data/feature', show_result=False):
    # Load the dataset
    print("Loading data...")
    test_only = ( model_file and os.path.isfile(model_file) )
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(training_dir=training_dir,
                                                                  testing_dir=testing_dir, 
                                                                  test_only=test_only)
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.imatrix('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    print('building network -- ', network_model)

    network = Han_cnn(input_var)


    print('defining loss function')
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    print('defining update')
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.005, momentum=0.9)

    print('defining testing method')
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_pred_int = T.cast(T.round(test_prediction), 'int32')
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                       target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:

    print('compiling training function')
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print('compiling validation function')
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_pred_int])


    if not model_file or not os.path.isfile(model_file):
        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        lowest_err = 100000.0
        temp_filename = '.temp_model_{}.npy'.format(int(time.time()*100000))
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, PD[BATCH_SIZE], shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # train_windows = 0
            # for clip, target in zip(X_train, y_train):
            #     # window size default to 1*20480/512
            #     for window in iter_windows(clip, 40):
            #         train_err += train_fn(window, target)
            #         train_windows += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            out_list, tar_list = [], []
            for batch in iterate_minibatches(X_val, y_val, PD[BATCH_SIZE], shuffle=False):
                inputs, targets = batch
                err, out  = val_fn(inputs, targets)
                val_err += err
                val_acc += f1_score(targets, out, average='micro')
                val_batches += 1
            
            # val_windows = 0
            # for clip, target in zip(X_val, y_val):
            #     # window size default to 1*20480/512
            #     for window in iter_windows(clip, 40):
            #         err, acc = val_fn(window, target)
            #         val_err += err
            #         val_acc += acc
            #         val_windows += 1


            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            if (val_err/val_batches) < lowest_err:
                lowest_err = val_err / val_batches
                final_param = lasagne.layers.get_all_param_values(network)
                np.save(temp_filename, final_param)

        # return to the best weights
        val = np.load(temp_filename)
        lasagne.layers.set_all_param_values(network, val)
        os.remove(temp_filename)


        if model_file:
            final_param = lasagne.layers.get_all_param_values(network)
            np.save(model_file, final_param)

    else:
        print('Load weights from file {}'.format(model_file))

        val = np.load(model_file)
        lasagne.layers.set_all_param_values(network, val)
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, PD[BATCH_SIZE], shuffle=False):
        inputs, targets = batch
        err, out  = val_fn(inputs, targets)
        test_err += err
        test_acc += f1_score(targets, out, average='micro')
        test_batches += 1
        if show_result:
            for k in range(len(targets)):
                print('target: {}'.format(targets[k]))
                print('output: {}'.format(out[k]))
                print('')
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

def usage():
    print("Instrument Auto-tagging using Convolutional Neural Network")


if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv[1:]) > 0:
        
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hsp:e:m:n:l:t:",
                                 ["help", "show_result_data", "parameter", "epoch", "model", "network", "training", "testing"])
        except getopt.GetoptError as err:
            print(str(err))
            usage()
        for o, a in opts:
            if o in ('-h', '--help'):
                usage()
                sys.exit()
            elif o in ('-s', '--show_result_data'):
                kwargs['show_result'] = True
            elif o in ('-p', '--parameter'):
                kwargs['param_file'] = a
            elif o in ('-e', '--epoch'):
                kwargs['num_epochs'] = int(a)
            elif o in ('-m', '--model'):
                kwargs['model_file'] = a
            # elif o in ('-n', '--network'):
            #     kwargs['network_model'] = a
            elif o in ('-l', '--training'):
                kwargs['training_dir'] = a
            elif o in ('-t', '--testing'):
                kwargs['testing_dir'] = a

    

    main(**kwargs)    
    
        
