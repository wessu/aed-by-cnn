import numpy as np
import utils
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import os

class Fresult:
	def __init__(self, name, target,
				 pred, g_pred, pg_pred,
				 annotation=None):
		'''
		pred:
			Final result of clip level prediction 
			shape = (10,)

		g_pred: 
			Final result of frame level prediction
			shape = (10, 3, None)

		pg_pred:
			The result of frame level prediction before Gaussian layer
			shape = (10, None)
		'''

		# up-sampling
		# g_pred = np.repeat(g_pred, 16, axis=-1)
		# pg_pred = np.repeat(pg_pred, 16, axis=-1)

		# self.annotation = utils.load_annotation(name, np.argmax(target))

		self.annotation = annotation

		self.name = name
		self.target = target
		self.pred = pred
		self.g_pred = g_pred
		self.pg_pred = pg_pred
		self.frame2time = 0.01161
		self.class_list = ['air_conditioner', 
		                  'car_horn', 
		                  'children_playing', 
		                  'dog_bark',
		                  'drilling',
		                  'engine_idling',
		                  'gun_shot',
		                  'jackhammer',
		                  'siren',
		                  'street_music']



	def isCorrectResult(self):
		return np.argmax(self.target) == np.argmax(self.pred)

	def plot_prediction(self, y, l='', label='', x_len=None):
		# x = np.asarray([xx*0.01161 for xx in x])
		x = np.array([xx*self.frame2time for xx in range(len(y))])
		y = np.array(y)
		line, = plt.plot(x, y, l, label=label)
		ax = plt.gca()
		ax.set_ylim((0, 1.1))
		if x_len == None: x_len = len(x)*self.frame2time
		ax.set_xlim((0, x_len))
		return line, ax

	def plotFrameLevelResult(self, img_dir, plot_gp=True, plot_pgp=True):
		plt.figure(1)
		for ii, [gp, pgp, anno] in enumerate(zip(self.g_pred, self.pg_pred, self.annotation)):
			plt.subplot(4, 3, ii+1)
			title = str(ii) if ii != np.argmax(self.pred) else '{} (predicted)'.format(ii)
			plt.title(title)
			plt.xlabel('Time')
			plt.ylabel('Prediction')
			plt.grid(True)
			line, ax = self.plot_prediction(anno, 'r')
			_x = np.array([xx*self.frame2time for xx in range(len(anno))])
			ax.fill_between(_x, anno, 0, facecolor='#E5E5E5')
			# for p, c in zip(gp, ['b', 'g', 'y']):
			# 	self.plot_prediction(p, c)
			if plot_pgp: self.plot_prediction(pgp, 'g')
			if plot_gp: self.plot_prediction(gp, 'b')
			
		plt.tight_layout()
		plt.savefig(os.path.join(img_dir, self.name))
		plt.close()

	def plotOneResult(self, img_dir, class_num, plot_gp=True, plot_pgp=False, plot_anno=False, x_len=None, fs=12):
		plt.figure(1)
		gp, pgp, anno = self.g_pred[class_num], self.pg_pred[class_num], self.annotation[class_num]
		title = '{} - {}'.format(self.name, self.class_list[class_num])
		# if class_num == np.argmax(self.target): title += ' (answer)'
		# plt.title(title)
		plt.xlabel('Time (s)', fontsize=fs)
		plt.ylabel('Prediction', fontsize=fs)
		ls = []
		if plot_anno:
			line, ax = self.plot_prediction(anno, 'r', label='ground-truth', x_len=x_len)
			_x = np.array([xx*self.frame2time for xx in range(len(anno))])
			ax.fill_between(_x, anno, 0, facecolor='#E5E5E5')
			ls.append(line)
		if plot_pgp: 
			line, ax = self.plot_prediction(pgp, 'g', label='pre-Gaussian layer', x_len=x_len)
			ls.append(line)
		if plot_gp: 
			line, ax = self.plot_prediction(gp, 'b', label='Gaussian layer', x_len=x_len)
			ls.append(line)
		# plt.legend(handles=ls, mode="expand", ncol=3)
		# plt.tick_params(axis='both', which='major', labelsize=28)
		fp = os.path.join(img_dir, self.name)
		if not os.path.exists(fp): os.makedirs(fp)
		plt.tight_layout()
		plt.savefig(os.path.join(img_dir, self.name, str(class_num)))
		plt.close()


def plot_frame(fr):
	fr.plotFrameLevelResult(img_dir, plot_gp, plot_pgp)
	print(fr.name)

def initProcess(imgd, gp, pgp):
	global img_dir
	global plot_gp
	global plot_pgp
	img_dir = imgd
	plot_gp = gp
	plot_pgp = pgp

def show_all_result(res_list, img_dir, plot_gp=True, plot_pgp=False):
	from multiprocessing import Pool
	n_cores = 10
	img_dir = os.path.join('img', img_dir)
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)
	func = plot_frame
	pool = Pool(processes=n_cores,
				initializer=initProcess, 
				initargs=(img_dir, plot_gp, plot_pgp))
	result = pool.map(func, res_list)
	pool.close()
	pool.join()

def load_result(fp):
	a = np.load(fp)
	res_list = []
	for b in a:
		f = Fresult(**b)
		res_list.append(f)
	return np.array(res_list)

# def majorityPrediction(res_list):


def meanPrediction(res_list):
	result_map = np.zeros((10,10), dtype=np.int32)
	acc = 0.0
	for res in res_list:
		mean_pred = np.argmax(np.mean(res.g_pred, axis=1, keepdims=True))
		ans = np.argmax(res.target)
		result_map[ans][mean_pred] += 1
		if mean_pred == ans: 
			acc += 1.0
	print("{} out of {} are correct.".format(acc, len(res_list)))
	print("mean pooling accuracy:\t\t{:.2f} %".format(
		acc / len(res_list) * 100))
	print("Result map: (x: prediction, y: target)")
	print(result_map)
	np.set_printoptions(precision=3)
	each_acc = np.array([float(result_map[i][i]) / np.sum(result_map[i]) * 100 for i in range(len(result_map))])
	print(each_acc)
	
def maxPrediction(res_list):
	result_map = np.zeros((10,10), dtype=np.int32)
	acc = 0.0
	for res in res_list:
		mean_pred = np.argmax(np.amax(res.g_pred, axis=1, keepdims=True))
		ans = np.argmax(res.target)
		result_map[ans][mean_pred] += 1
		if mean_pred == ans: 
			acc += 1.0
	print("{} out of {} are correct.".format(acc, len(res_list)))
	print("mean pooling accuracy:\t\t{:.2f} %".format(
		acc / len(res_list) * 100))
	print("Result map: (x: prediction, y: target)")
	print(result_map)
	np.set_printoptions(precision=3)
	each_acc = np.array([float(result_map[i][i]) / np.sum(result_map[i]) * 100 for i in range(len(result_map))])
	print(each_acc)

def auc_score(res_list):
	gp_list = np.array([])
	anno_list = np.array([])
	for res in res_list:
		g_pred = res.g_pred
		anno = res.annotation
		if g_pred.shape[-1] < anno.shape[-1]:
			anno = np.delete(anno, range(g_pred.shape[-1], anno.shape[-1]), axis=-1)
		elif g_pred.shape[-1] > anno.shape[-1]:
			g_pred = np.delete(g_pred, range(anno.shape[-1], g_pred.shape[-1]), axis=-1)
		gp_list = g_pred.T if len(gp_list) == 0 else np.append(gp_list, g_pred.T, axis=0)
		anno_list = anno.T if len(anno_list) == 0 else np.append(anno_list, anno.T, axis=0)

	assert(gp_list.shape == anno_list.shape)

	from sklearn.metrics import roc_auc_score
	class_auc = roc_auc_score(anno_list, gp_list, average=None)
	print('AUC of Classes:')
	print(class_auc)
	all_micro_auc = roc_auc_score(anno_list, gp_list, average='micro')
	print('Total micro AUC: {}'.format(all_micro_auc))

	all_macro_auc = roc_auc_score(anno_list, gp_list, average='macro')
	print('Total macro AUC: {}'.format(all_macro_auc))

def acc_score(res_list):
	acc_list = np.zeros(10, dtype=float)
	num = np.zeros(10, dtype=np.int32)
	for res in res_list:
		tar = np.argmax(res.target)
		if np.argmax(res.pred) == tar:
			acc_list[tar] += 1
		num[tar] += 1

	total_acc = float(np.sum(acc_list))/np.sum(num) * 100
	print(total_acc)
	acc = np.divide(acc_list*100, num)
	print(acc)


for i in range(0, 40, 10):
	fn0 = 'var/ftd/train_' + str(i) + '.npy'
	bats = np.load(fn0)
	print(len(bats))
	for j in range(i+1, 1+10):
		fn = 'var/ftd/train_' + str(j) + '.npy'
		bats = np.append(bats, np.load(fn))
		print('i: {}, j:{}, len_bats:{}'.format(i, j, len(bats)))
	fnf = 'var/stdfeat/train_' + str(i) + '.npy'
	np.save(fnf, bats)
	print('{} saved'.format(fnf))

