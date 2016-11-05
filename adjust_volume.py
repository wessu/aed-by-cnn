import sys
import os
from os import path
import librosa
import pydub
from pydub import AudioSegment as aseg 

def adjust_volume(in_fp):
	def adjust(volume):
		audio_p = audio + volume
		fn_p = fn + "_" + str(volume) +"db" + ".wav"
		fd = audio_p.export(path.join(out_dir, str(volume) + 'db', path.split(in_dir)[-1], fn_p), format=format)

	in_dir, fn = path.split(in_fp)
	fn, file_ext = path.splitext(fn)
	file_ext = file_ext.lower()
	format = file_ext.replace('.', '')
	# audio = None
	y, sr = librosa.load(in_fp, sr=44100)
	tmp_in_fp = "tmp/" + fn + "_tmp.wav"
	librosa.output.write_wav(tmp_in_fp, y, sr, norm=False)
	format = "wav"
	audio = aseg.from_file(tmp_in_fp, format)
	os.remove(tmp_in_fp)
	
	if audio != None:
		for v in volume_list:
			adjust(v)

def initProcess(od, vl):
	global out_dir
	global volume_list
	out_dir = od
	volume_list = vl

def main(src_dir, out_dir):
	fp_list = []
	class_dir = []
	for root, dirs, files in os.walk(src_dir):
		if len(files) > 0:
			nof = 0
			for in_fn in files:
				if not (in_fn.endswith('.csv') or 
						in_fn.endswith('.json') or 
						in_fn.startswith('.')):
					in_fp = path.join(root, in_fn)
					fp_list.append(in_fp)
					cls = path.split(root)[-1]
					if cls not in class_dir:
						class_dir.append(cls)
					nof += 1
			print('{} files in {}'.format(nof, root))
	print('We got totally {} files.'.format(len(fp_list)))

	volume_list = [5, -5]
	
	for volume in volume_list:
		for cl in class_dir:
			vol_dir = path.join(out_dir, str(volume) + 'db', cl)
			if not path.exists(vol_dir):
				os.makedirs(vol_dir)

	func = adjust_volume
	n_cores = 10
	from multiprocessing import Pool
	pool = Pool(processes=n_cores, 
				initializer=initProcess, 
				initargs=(out_dir, volume_list))
	result = pool.map(func, fp_list)
	pool.close()
	pool.join()


if __name__ == '__main__':
	src_dir = sys.argv[1]
	out_dir = sys.argv[2]
	main(src_dir, out_dir)