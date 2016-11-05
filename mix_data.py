import numpy as np 
from fex_test import extractFeatures as fex


def gen_8k_dict(dirs):
	feat_dict = {}
	for src_dir, feat_type in dirs:
		all_feat, feat_dir = fex(src_dir, dataset='8k')
		print('all_feat length: {}'.format(len(all_feat)))
		for feat in all_feat:
			name = str(feat[-1]).split('_')[0]
			if name not in feat_dict:
				feat_dict[name] = {}
			feat_dict[name][feat_type] = feat

	return feat_dict

def gen_us_dict(dirs):
	feat_dict = {}
	for src_dir, feat_type in dirs:
		all_feat, feat_dir = fex(src_dir, dataset='us')
		for feat in all_feat:
			name = str(feat[-1]).split('_')[0]
			if name not in feat_dict:
				feat_dict[name] = {}
			feat_dict[name][feat_type] = feat

	return feat_dict

def genDict(dirs, dataset):
	feat_dict = {}
	for src_dir, feat_type in dirs:
		all_feat, feat_dir = fex(src_dir, dataset=dataset)
		for feat in all_feat:
			name = str(feat[-1]).split('_')[0]
			if name not in feat_dict:
				feat_dict[name] = {}
			feat_dict[name][feat_type] = feat

	return feat_dict

# def main():


if __name__ == '__main__':
	dirs = [('/home/dovob/NAS/Database/UrbanSound8K/audio', 'origin'),
			('data/audio/augment/5db', '5db'),
			('data/audio/augment/-5db', '-5db')]
	feat_dict = gen_us_dict(dirs)
	print('feat_dict length: {}'.format(len(feat_dict.keys())))
	for fn in feat_dict:
		np.save('data/feature/US_Dict/'+fn+'.npy', feat_dict[fn])

	# np.save('data/feature/8K_Dict.npy', feat_dict)
