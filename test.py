import os
from os import path
from multiprocessing import Pool


def scan(scan_dir):
	src_dir = '/home/dovob/NAS/Database/UrbanSound/data'
	train_dir = 'data/feature/logmelspec10000.44100_1024_512_128.0.raw/training'
	test_dir = 'data/feature/logmelspec10000.44100_1024_512_128.0.raw/testing'
	print('scanning {}...'.format(scan_dir))
	for root, dirs, files in os.walk(path.join(src_dir, scan_dir)):
		# print('scanning {}...'.format(root))
		if files:
			for fi in files:
				if fi.endswith('.csv'):
					fn, ext = path.splitext(fi)
					if not (path.exists(path.join(train_dir, scan_dir, fn + '.npy')) or
							path.exists(path.join(test_dir, scan_dir, fn + '.npy'))):
						print('{}/{} is not extracted.'.format(scan_dir, fi))


if __name__ == '__main__':

	func = scan
	dirs = ['air_conditioner',  'car_horn',  'children_playing',  'dog_bark',  'drilling',
			'engine_idling',  'gun_shot',  'jackhammer',  'siren',  'street_music']
	pool = Pool(processes=10)
	result = pool.map(func, dirs)
	pool.close()

