import os
import sys
import librosa
import numpy as np
from os import path
from random import sample
from multiprocessing import Lock
from multiprocessing import Manager
import time

TOTAL_NUM_TAGS = 10

PAD_TO_SIZE = 500


tag_dict = {
    'air_conditioner' : 0,
    'car_horn' : 1,
    'children_playing' : 2,
    'dog_bark' : 3,
    'drilling' : 4,
    'engine_idling' : 5,
    'gun_shot' : 6,
    'jackhammer' : 7,
    'siren' : 8,
    'street_music' : 9,
}

def tag_of_file(filename, typename):
    tag = np.zeros(len(tag_dict), dtype=int)
    tag[tag_dict[typename]] = 1
    return tag

def ef_log_10000(args):
    in_fp, feat_dir, sr, win_size, hop_size, n_mels, file_ext = args
    in_dir, fn = path.split(in_fp)
    fn = fn.replace(file_ext, '')
    out_dir = path.join(feat_dir, path.split(in_dir)[-1])

    out_fp = path.join(out_dir, '{}.npy'.format(fn))

    def extract(duration=None):
        # start_time = time.time()
        sig = librosa.core.load(in_fp, sr=sr, mono=True, duration=duration)
        feat_ = librosa.feature.melspectrogram(sig[0], sr=sr,
                                               n_fft=win_size,
                                               hop_length=hop_size,
                                               n_mels=n_mels).T
        # Pad zeros to hundred digits
        npad = ((0, PAD_TO_SIZE - feat_.shape[0]%PAD_TO_SIZE), (0, 0))
        feat_ = np.pad(feat_, pad_width=npad, 
        			   mode='constant', constant_values=0)

        # logarithmize
        feat_ = np.array([np.log(1+10000*feat_)], dtype=np.float32)
        target = tag_of_file(fn, path.split(in_dir)[-1])
        
        lock.acquire()
        all_feat.append((feat_, target))

        # np.save(out_fp, (feat_, target))
        # print('{} extracted. duration: {:.3f}s'.format(in_fp, time.time()-start_time))
        lock.release()

    try:
        extract()

    except Exception as err:
        print('{} extraction failed. Error: {}'.format(in_fp, err))
        extract(duration=39.7)
        print('{} extracted!'.format(in_fp))

def feat_2_data_dict(feat_list):
	_data = {}
	for feat in feat_list:
		feat_len_str = str(feat[0].shape[1])
		if feat_len_str not in _data:
			_data[feat_len_str] = np.asarray([feat])
			# print('_data get {}. {}'.format(feat_len_str, _data[feat_len_str].shape))
		else:
			_data[feat_len_str] = np.append(_data[feat_len_str], [feat], axis=0)
			# print('_data[{}] shape: {}'.format(feat_len_str, _data[feat_len_str].shape))

	# print sizes
	for d in _data:
		print('{}:\t{}'.format(d, len(_data[d])))
	return _data

def initProcess(af, l):
	global all_feat
	global lock
	all_feat = af
	lock = l

def setup(src_dir, base_dir=''):
    # base_dir = '/ABC'
    # src_dir = '/ABC'

    if base_dir is '':
        base_dir = path.split(path.abspath(src_dir))[0]
    n_cores = 10

    feat_type = 'logmelspec10000'
    sr = 44100
    win_size = 1024
    hop_size = 512
    n_mels = 128
    
    base_feat_dir = path.join(base_dir, 'feature')
    feat_dir = path.join(base_feat_dir, '{}.{}_{}_{}_{}.{}.raw'.format(
        feat_type,
        sr,
        win_size,
        hop_size,
        n_mels,
        0))
    train_dir = path.join(feat_dir, 'training')

    if not path.exists(train_dir):
        os.makedirs(train_dir)

    fp_list = []
    for root, dirs, files in os.walk(src_dir):
        if files:
            nof = 0
            for in_fn in files:
                if not (in_fn.endswith('.csv') or in_fn.endswith('.json') or in_fn.startswith('.')):
                    in_fp = path.join(root, in_fn)
                    fp_list.append(in_fp)
                    out_dir = path.join(train_dir, path.split(root)[-1])
                    if not path.exists(out_dir):
                        os.makedirs(out_dir)
                    nof += 1
            print('{} files in {}'.format(nof, root))


    print('We got {} files.'.format(len(fp_list)))
    

    all_feat = Manager().list()
    lock = Lock()
    # for term in range(10)+['a', 'b', 'c', 'd', 'e', 'f']:
    #     temp_dir = path.join(feat_dir, str(term))
    #     if not path.exists(temp_dir):
    #         os.makedirs(temp_dir)

    func = ef_log_10000

    arg_list = []
    for in_fp in fp_list:
        arg_list.append(
            (in_fp, train_dir, sr, win_size, 
             hop_size, n_mels, path.splitext(in_fp)[-1]))
    from multiprocessing import Pool
    pool = Pool(processes=n_cores, initializer=initProcess, initargs=(all_feat, lock))
    result = pool.map(func, arg_list)
    pool.close()
    pool.join()

    np.save('all_feat.npy', all_feat)

    # Chop some as testing data
    np.random.shuffle(all_feat)
    chop = int(len(all_feat)/5)
    # test_feat = all_feat[:chop]
    # all_feat = all_feat[chop:]

    # Save training data
    train_data = feat_2_data_dict(all_feat[chop:])
    tr_out_fp = path.join(feat_dir, 'train_data.npy')
    np.save(tr_out_fp, train_data)

    # Save testing data
    test_data = feat_2_data_dict(all_feat[:chop])
    te_out_fp = path.join(feat_dir, 'test_data.npy')
    np.save(te_out_fp, test_data)

    return


    # print('Randomly choosing testing data...')
    # test_dir = path.join(feat_dir, 'testing')
    # if not path.exists(test_dir):
    #     os.makedirs(test_dir)
    # for root, dirs, files in os.walk(train_dir):
    # 	if files:
    # 		chosen_f = sample(files, int(round(len(files)/5)))
    # 		for cf in chosen_f:
    # 			new_path = path.join(test_dir, path.basename(root))
    # 			if not path.exists(new_path):
    # 			    os.makedirs(new_path)
    # 			os.rename(path.join(root, cf), path.join(new_path, cf))
    # 		print('{} files of {} are chosen to be testing files.'.format(len(chosen_f), path.basename(root)))




if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or len(sys.argv) < 3:
        print('')
        print("Extracts the melspectrograms from files.")
        print("Usage: %s [SRC_DIR [BASE_DIR]]" % sys.argv[0])
        print('')
        print("       SRC_DIR: the directory of the source files.")
        print("       BASE_DIR: the base directory of the output files.")
        print('')
        print('')

    else:
        src_dir = sys.argv[1]
        if not path.exists(src_dir):
            print('path not exists! {}'.format(src_dir))
        
        else:
            base_dir = sys.argv[2]
            if not path.exists(base_dir):
                os.makedirs(base_dir)
            st = time.time()
            setup(src_dir, base_dir)
            print('Total duration: {:.3f}s'.format(time.time()-st))