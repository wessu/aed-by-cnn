import numpy as np

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

def main():
	all_feat = np.load('all_feat.npy').tolist()
	# Chop some as testing data
	np.random.shuffle(all_feat)
	chop = int(len(all_feat)/5)
	test_feat = all_feat[:chop]

	# Save testing data
	test_data = feat_2_data_dict(test_feat)
	np.save('test_data.npy', test_data)
	
	del all_feat[:chop]
	# Save training data
	train_data = feat_2_data_dict(all_feat)
	np.save('train_data.npy', train_data)

	return

if __name__ == '__main__':
	main()