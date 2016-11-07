import argparse
import numpy as np
import os
from sklearn import preprocessing as pp
import time
from utils import my_iterator
from multiprocessing import Pool
import threading

US_Dict_filepath = 'data/feature/US_Dict'
EK_Dict_filepath = 'data/feature/8K_Dict'
TW_Dict_filepath = 'data/feature/TW_Dict'
global delta
global augment

##### UTILITY FUNCTIONS #####
def getDictDir(dict_type):
    if dict_type == 'us':
        return US_Dict_filepath
    elif dict_type == 'tw':
        return TW_Dict_filepath
    else:
        return EK_Dict_filepath

def add_feature(data_list, ft_dict, name):
    if name in ft_dict:
        ft = ft_dict[name]
        if not delta:
            ft = ([np.delete(k, 1, 0) for k in ft[0]], ft[1], ft[2])
        data_list.append(ft)

def load_files(data_dir, fn_list):
    data_list = []
    for fn, fl_type in fn_list:
        ft_dict = np.load(os.path.join(data_dir, fn+'.npy')).item()
        add_feature(data_list, ft_dict, fl_type)
    return data_list

def fit_scaler(sc_list, data_list):
    ipts, tars, names = zip(*data_list)
    if len(sc_list) == 0:
        print('Creating scaler...')
        for _x in zip(*ipts):
            sc_list.append(pp.StandardScaler())
        print('Created a scaler with length of {}'.format(len(sc_list)))
    for x, scaler in zip(zip(*ipts), sc_list):
        len_list = [l.shape[1] for l in x]
        z = []
        for p in x:
            for q in p[0]:
                z.append(q)
        scaler.partial_fit(z)

def standardize(sc_list, data_list):
    ipts, tars, names = zip(*data_list)
    for ipt in ipts:
        for feat, scaler in zip(ipt, sc_list):
            feat[0] = scaler.transform(feat[0])

##### MULTITHREADING #####
threadLock = threading.Lock()

def printMsg(msg):
    threadLock.acquire()
    print(msg)
    threadLock.release()

class myThread(threading.Thread):
    def __init__(self, name, load_dir, fl_list, sc_list, save_dir):
        threading.Thread.__init__(self)
        self.name = name
        self.load_dir = load_dir
        self.fl_list = fl_list
        self.sc_list = sc_list
        self.save_dir = save_dir
    def run(self):
        printMsg("Starting " + self.name) 
        num_fl = 0
        args_list = []
        for fn_list, idx in my_iterator(self.fl_list, 500):
            args_list.append((fn_list, idx, self.load_dir, 
                         self.sc_list, self.save_dir, self.name))
            num_fl += len(fn_list)
        printMsg('{} data: {}'.format(self.name, num_fl))
        std_by_multiprocessing(args_list)

        

##### MULTIPROCESSING #####
def std_by_multiprocessing(args_list):
    pool = Pool(processes=6)
    pool.map(standardize_list, args_list)
    pool.close()
    pool.join()

def standardize_list(args):
    fn_list, idx, load_dir, sc_list, save_dir, name = args
    data_list = load_files(load_dir, fn_list)
    standardize(sc_list, data_list)
    ft_batches = np.asarray([data_list[i:i+1] for i in range(0, len(data_list), 1)])
    save_fp = os.path.join(save_dir, name+'_'+str(idx)+'.npy')
    np.save(save_fp, ft_batches)
    printMsg('{} finished.'.format(name))


##### MAIN FUNCTION #####
def load_dataset(training_type, testing_type, test_fold, scalers_file):
    train_dir = getDictDir(training_type)
    test_dir = getDictDir(testing_type)
    test_only = ( scalers_file and os.path.isfile(scalers_file) )

    # select testing files
    if not os.path.isdir(test_fold):
        test_fold = '/home/dovob/NAS/Database/UrbanSound8K/audio/fold' + str(test_fold)
    assert(os.path.isdir(test_fold))
    test_file_set = set()
    for root, dirs, files in os.walk(test_fold):
        for fn in files:
            fn = os.path.splitext(fn)[0]
            test_file_set.add(fn)
            fid = fn.split('-')[0]
            test_file_set.add(fid)

    if test_only:
        save_dir = 'var/stdfeat/' + os.path.basename(scalers_file).split('.')[0]
    elif scalers_file != None:
        save_dir = 'var/stdfeat/' + scalers_file
    else:
        save_dir = 'var/stdfeat/tmp{}'.format(int(time.time()*100000))
    print('save_dir: {}'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    threads = []

    if test_only:
        # load scalers to standardize testing files
        print('Loading testing files only')
        sc_fp = 'model/scalers/scaler_' + os.path.basename(scalers_file)
        sc_list = np.load(sc_fp).tolist()
    else:
        # select training and validation files
        tmp_tv_list = []
        for fn in os.listdir(train_dir):
            fn = os.path.splitext(fn)[0]
            if fn not in test_file_set:
                tmp_tv_list.append(fn)
        
        np.random.shuffle(tmp_tv_list)
        tv_list = []
        if augment:
            for fn in tmp_tv_list:
                tv_list.append((fn, 'origin'))
                tv_list.append((fn, '5db'))
                tv_list.append((fn, '-5db'))
        else:
            for fn in tmp_tv_list:
                tv_list.append((fn, 'origin'))
        print('Totally {} files for training and validation.'.format(len(tv_list)))

        np.random.shuffle(tv_list)
        sp = int(len(tv_list)/5)
        print('{} files for va'.format(sp))
        va_list = tv_list[:sp]
        tr_list = tv_list[sp:]

        # create scalers
        sc_list = []
        for fn_list, idx in my_iterator(tr_list, 1000):
            data_list = load_files(train_dir, fn_list)
            fit_scaler(sc_list, data_list)
            print('{} files to the scalers.'.format(len(data_list)))

        # standardize training and validation data and save them
        tr_thread = myThread("train", train_dir, tr_list, sc_list, save_dir)
        va_thread = myThread("val", train_dir, va_list, sc_list, save_dir)
        tr_thread.start()
        va_thread.start()
        threads.append(tr_thread)
        threads.append(va_thread)

        if scalers_file != None:
            np.save('model/scalers/scaler_' + scalers_file, sc_list)
        
        # num_tr = 0
        # for fn_list, idx in my_iterator(tr_list, 500):
        #     data_list = load_files(train_dir, fn_list)
        #     standardize(sc_list, data_list)
        #     tr_batches = np.asarray([data_list[i:i+1] for i in range(0, len(data_list), 1)])
        #     np.save(save_dir+'/train_'+str(idx)+'.npy', tr_batches)
        #     num_tr += len(data_list)
        # print('train data: {}'.format(num_tr))

        # num_va = 0
        # for fn_list, idx in my_iterator(va_list, 500):
        #     data_list = load_files(train_dir, fn_list)
        #     standardize(sc_list, data_list)
        #     va_batches = np.asarray([data_list[i:i+1] for i in range(0, len(data_list), 1)])
        #     np.save(save_dir+'/val_'+str(idx)+'.npy', va_batches)
        #     num_va += len(data_list)
        # print('val data: {}'.format(num_va))

    # standardize testing and save them
    te_list = []
    for fn in os.listdir(test_dir):
        fn = os.path.splitext(fn)[0]
        if fn in test_file_set:
            te_list.append((fn, 'origin'))

    te_thread = myThread("test", test_dir, te_list, sc_list, save_dir)
    te_thread.start()
    threads.append(te_thread)


    # for data_list, idx in my_iterator(te_data, 500):
    #     standardize(sc_list, data_list)
    #     te_batches = np.asarray([data_list[i:i+1] for i in range(0, len(data_list), 1)])
    #     np.save(save_dir+'/test_'+str(idx)+'.npy', te_batches)
    # print('test data: {}'.format(len(te_data)))

    for t in threads:
        t.join()

def parser():
    p = argparse.ArgumentParser(description="Audio Event Detection using Fully Convolutional Network")
    p.add_argument('-l', '--training', type=str, metavar='training_type',
                    help='Training type')
    
    p.add_argument('-t', '--testing', type=str, metavar='testing_type',
                    help='Testing type')
    
    # p.add_argument('-r', '--rd', action='store_true', default=False,
    #                 help='Show result data')
    
    # p.add_argument('-e', '--epoch', type=int, 
    #                 help='Number of epochs')
    
    # p.add_argument('-s', '--scales', type=int, default=3,
    #                 help='Number of scales (1, 2, or 3)')
    
    p.add_argument('-s', '--scalers', type=str, 
                    help='The scaler file')
    
    # p.add_argument('-s', '--std', action='store_true', default=True,
    #                 help='Standardize data')
    
    # p.add_argument('-f', '--frame', type=str,
    #                 help='Output frame level prediction')
    
    # p.add_argument('-c', '--continue_train', action='store_true', default=False,
    #                 help='Continue to train a model')

    p.add_argument('-F', '--fold', type=str,
                    help='Test folder')

    p.add_argument('-a', '--augment', action='store_true', default=False,
                    help='Train with data augmentation')

    # p.add_argument('-g', '--gaussian', action='store_true', default=False,
    #                 help='Add gaussian layer to model')

    p.add_argument('-d', '--delta', action='store_true', default=False,
                    help='account delta as feature')


    args = p.parse_args()
    return args


def usage():
    print("Generate Standardized Features")


if __name__ == '__main__':
    kwargs = {}
    args = parser()
    kwargs['scalers_file'] = args.scalers
    kwargs['training_type'] = args.training
    kwargs['testing_type'] = args.testing
    kwargs['test_fold'] = args.fold
    
    augment = args.augment
    delta = args.delta

    print('training data: {}'.format(kwargs['training_type']))
    print('testing data: {}'.format(kwargs['testing_type']))
    print('scalers file: {}'.format(kwargs['scalers_file']))
    print('test fold: {}'.format(kwargs['test_fold']))

    load_dataset(**kwargs)
    