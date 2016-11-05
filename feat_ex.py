import os
import sys
import librosa
import numpy as np
from os import path
from random import sample
from multiprocessing import Lock
from multiprocessing import Manager
from multiprocessing import Pool
import time

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
MIN_LEN = 180

def _fex(in_fp, tag_of_file):
	def ef_log_10000(in_fp):
		def npad(ol):
			x = MIN_LEN - ol if (ol < MIN_LEN) else 0
			return ((0, x), (0, 0))
		try:
			sig = librosa.core.load(in_fp, sr=sr, mono=True)
			feat_list = []
			for ws in win_size:
				ft = librosa.feature.melspectrogram(sig[0], sr=sr,
				                                   n_fft=ws,
				                                   hop_length=hop_size,
				                                   n_mels=n_mels).T
				if ft.shape[0] < MIN_LEN:
					ft = np.pad(ft, pad_width=npad(ft.shape[0]), 
								mode='constant', constant_values=0)
				ft = np.log(1+10000*ft)
				delta = librosa.feature.delta(ft, axis=1)

				# logarithmize
				feat_list.append(np.array([ft, delta], dtype=np.float32))
			return feat_list
		except Exception as err:
			print('{} extraction failed. Error: {}'.format(in_fp, err))
			return None

	start_time = time.time()
	fn, tn = getFileInfo(in_fp, dataset)
	features = ef_log_10000(in_fp)
	target = tag_of_file(tn)

	lock.acquire()
	all_feat.append((features, target, fn))
	# print('{} extracted. length: {}'.format(fn, len(features[0][0])))
	lock.release()

def getFileInfo(in_fp, dataset):
	target_number = None
	file_name = None
	if dataset == '8k':
		in_dir, fn = path.split(in_fp)
		file_ext = path.splitext(in_fp)[-1]
		file_name = str(fn.replace(file_ext, ''))
		fl_info = file_name.split('-')
		# origin_name = fl_info[0]
		target_number = int(fl_info[1])
	elif dataset == 'us':
		in_dir, fn = path.split(in_fp)
		file_ext = path.splitext(in_fp)[-1]
		file_name = str(fn.replace(file_ext, ''))
		target_number = int(tag_dict[path.split(in_dir)[-1]])

	assert(type(file_name) == str)
	assert(type(target_number) == int)

	return file_name, target_number

def _func_tag_label(in_fp):
	def tag_of_file(tag_msg):
		_tag = np.zeros(NUM_TAG, dtype=int)
		_tag[tag_msg] = 1
		return _tag

	_fex(in_fp, tag_of_file)

def _func_tag_class(in_fp):
	def tag_of_file(tag_msg):
		return tag_msg

	_fex(in_fp, tag_of_file)

def isAudioFile(fn, dataset):
	if dataset == '8k':
		return fn.endswith('.wav')
	elif dataset == 'us':
		return not (fn.endswith('.csv') or 
					fn.endswith('.json') or 
					fn.startswith('.'))
	return True

def initProcess(ds, af, l, s, w, h, n):
	global dataset
	global all_feat
	global lock
	global sr
	global win_size
	global hop_size
	global n_mels
	global NUM_TAG
	dataset = ds
	all_feat = af
	lock = l
	sr = s
	win_size = w
	hop_size = h
	n_mels = n
	NUM_TAG = 10

def extractFeatures(src_dir, base_dir='extracted_features', dataset='us', save=False):
	if base_dir is '':
		base_dir = path.split(path.abspath(src_dir))[0]
	n_cores = 10

	feat_type = 'logmelspec10000'
	sr = 44100
	win_size = [1024, 4096, 16384]
	hop_size = 512
	n_mels = 128
	
	dir_name = '{}.{}_{}_{}'.format(feat_type, sr, hop_size, n_mels)
	for ws in win_size:
		dir_name += '.' + str(int(ws/1024))
	feat_dir = path.join('data/feature', 
						 dir_name,
						 base_dir)
	
	fp_list = []
	for root, dirs, files in os.walk(src_dir):
		if files:
			nof = 0
			for in_fn in files:
				if isAudioFile(in_fn, dataset):
					in_fp = path.join(root, in_fn)
					fp_list.append(in_fp)
					nof += 1
			print('{} files in {}'.format(nof, root))
	print('We got totally {} files.'.format(len(fp_list)))

	all_feat = Manager().list()
	test_file_list = Manager().list()
	lock = Lock()
	func = _func_tag_label
	
	pool = Pool(processes=n_cores, 
				initializer=initProcess, 
				initargs=(dataset, all_feat, lock, sr, 
						  win_size, hop_size, n_mels))
	result = pool.map(func, fp_list)
	pool.close()
	pool.join()
	if save:
		if not path.exists(feat_dir):
			os.makedirs(feat_dir)
		np.random.shuffle(all_feat)
		np.save(path.join(feat_dir, 'all_feat.npy'), all_feat)

	return all_feat, feat_dir


if __name__ == '__main__':
	kwargs = {}
	kwargs['src_dir'] = sys.argv[1]
	if not path.exists(src_dir):
		print('path not exists! {}'.format(src_dir))
	else:
		kwargs['base_dir'] = sys.argv[2]
		kwargs['dataset'] = sys.argv[3]
		kwargs['save'] = True
		st = time.time()
		extractFeatures(**kwargs)
		print('Total duration: {:.3f}s'.format(time.time()-st))
