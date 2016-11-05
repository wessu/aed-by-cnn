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
NUM_TAG = 10
out_dir = 'data/feature/TW_Dict/'

def _fex(in_fp, fn, target):
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
			print('ef_log_10000: {} extraction failed. Error: {}'.format(in_fp, err))
			return None

	if fn in all_feat:
		feat = np.load(out_dir+fn+'.npy').item()
	else:
		feat = {}
		# all_feat.append(fn)
		# start_time = time.time()
		# lock.acquire()
		# all_feat[fn] = (features, target, fn)
		# print('{} extracted. length: {}'.format(fn, len(features[0][0])))
		# lock.release()

	features = ef_log_10000(in_fp)
	feat[aug_type] = (features, target, fn)
	np.save(out_dir+fn+'.npy', feat)

def _func_tag_label(args):
	try:
		_fex(*args)
	except Exception as err:
		print('_func_tag_label: {} extraction failed. Error: {}'.format(args[0], err))

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

	file_name = file_name.split('_')[0]

	assert(type(file_name) == str)
	assert(type(target_number) == int)

	return file_name, target_number

# def _func_tag_class(in_fp):
# 	def tag_of_file(tag_msg):
# 		return tag_msg

# 	_fex(in_fp, tag_of_file)

def isAudioFile(fn, dataset):
	if dataset == '8k':
		return fn.endswith('.wav')
	elif dataset == 'us':
		return not (fn.endswith('.csv') or 
					fn.endswith('.json') or 
					fn.startswith('.'))
	return True

def initProcess(ds, af, l, s, w, h, n, ag):
	global dataset
	global all_feat
	global lock
	global sr
	global win_size
	global hop_size
	global n_mels
	global aug_type
	dataset = ds
	all_feat = af
	lock = l
	sr = s
	win_size = w
	hop_size = h
	n_mels = n
	aug_type = ag

def extractFeatures(src_dir, base_dir='extracted_features', dataset='us', aug_type='origin', save=False):
	if base_dir is '':
		base_dir = path.split(path.abspath(src_dir))[0]
	n_cores = 20

	feat_type = 'logmelspec10000'
	sr = 44100
	win_size = [1024, 4096, 16384]
	hop_size = 512
	n_mels = 128
	
	dir_name = '{}.{}_{}_{}'.format(feat_type, sr, hop_size, n_mels)
	for ws in win_size:
		dir_name += '.' + str(int(ws/1024))
	# feat_dir = path.join('data/feature', 
	# 					 dir_name,
	# 					 base_dir)
	
	fp_dict = {}
	fp_list = []
	for root, dirs, files in os.walk(src_dir):
		if files:
			nof = 0
			for in_fn in files:
				if isAudioFile(in_fn, dataset):
					in_fp = path.join(root, in_fn)
					fn, tn = getFileInfo(in_fp, dataset)
					if fn in fp_dict:
						fp_dict[fn][-1][tn] = 1
					else:
						target = np.zeros(NUM_TAG, dtype=int)
						target[tn] = 1
						fp_dict[fn] = (in_fp, fn, target)
					# fp_list.append(in_fp)
					nof += 1
			print('{} files in {}'.format(nof, root))
	print('We got totally {} clips.'.format(len(fp_dict.keys())))

	num_stat = np.zeros(NUM_TAG, dtype=int)
	for _p, _n, tags in fp_dict.values():
		num_stat[np.sum(tags)-1] += 1
	print('Number of clips according to each number of tags:')
	for i in range(NUM_TAG):
		print('tags: {}, clips: {}'.format(i, num_stat[i]))

	all_feat = Manager().list()
	test_file_list = Manager().list()
	lock = Lock()
	func = _func_tag_label

	for root, dirs, files in os.walk(out_dir):
		for _fn in files:
			fn = _fn.replace('.npy', '')
			all_feat.append(fn)

	# for f in all_feat:
	# 	if f in fp_dict:
	# 		del fp_dict[f]
	# print('{} clips left.'.format(len(fp_dict.keys())))
	# for f in fp_dict:
	# 	print(f)
	# return

	pool = Pool(processes=n_cores, 
				initializer=initProcess, 
				initargs=(dataset, all_feat, lock, sr, 
						  win_size, hop_size, n_mels, aug_type))
	result = pool.map(func, fp_dict.values())
	pool.close()
	pool.join()
	# if save:
	# 	if not path.exists(feat_dir):
	# 		os.makedirs(feat_dir)
	# 	np.save(path.join(feat_dir, 'all_feat.npy'), all_feat)

	# return all_feat, feat_dir

def concatAugments(dirs, save=False):
	for src_dir, aug_type in dirs:
		extractFeatures(src_dir, aug_type=aug_type)


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
