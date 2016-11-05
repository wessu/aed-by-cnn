import numpy as np
import urllib3 as ul
import json
import os

http = ul.PoolManager()

access_token = 'dy86lGqNclpPi9Mk5WYyD2iL6cDohM'

def search(target, tags, pack=None):
	filter_tags = 'tag:(' + ' OR '.join(tags) + ')'
	if pack != None:
		filter_tags += ' pack:\"{}\"'.format(pack)
	res_dict = {}
	def search_query(query, page=1):
		results = []
		print('--{}--, page {}'.format(query, page))
		res = http.request( 'GET',
							'http://www.freesound.org/apiv2/search/text/',
							headers={'Authorization': 'Bearer ' + access_token},
							fields={'query': query,
									'page_size': 150,
									'fields': 'id,url,name,tags,type',
									'filter': filter_tags,
									'page': page}
							)
		if res.status == 200:
			j = json.loads(res.data.decode('utf-8'))
			results = j['results']
			if j['next'] != None:
				results_n = search_query(query, page=(page+1))
				if results_n != None:
					results += results_n
			return results
		else:
			print('ERROR: {}'.format(res.status))
			print(json.loads(res.data.decode('utf-8')))
			return None
	if type(target) == type([]) or type(target) == type(np.array([])):
		for query in target:
			print('Searching {}...'.format(query))
			results = search_query(query)
			if results != None:
				print('Number of results: {}'.format(len(results)))
				for res in results:
					if res['id'] not in res_dict:
						res_dict[res['id']] = res
			else:
				print('Search of {} returned None'.format(query))
	elif type(target) == str:
		results = search_query(target)
		if results != None:
			for res in results:
				if res['id'] not in res_dict:
					res_dict[res['id']] = res
		else:
			print('Search of {} returned None'.format(query))
	print('Totally {} files.'.format(len(res_dict)))
	return res_dict

def _download_one(args):
	info, base_dir = args
	headers = {'Authorization': 'Bearer ' + access_token}
	idstr = str(info['id'])

	json_url = 'http://www.freesound.org/apiv2/sounds/' + idstr + '/'
	json_fp = os.path.join(base_dir, idstr + '.json')
	jres = http.request('GET',
						json_url,
						headers=headers)
	if jres.status == 200:
		jdata = json.loads(jres.data.decode('utf-8'))
		with open(json_fp, 'w') as f:
			json.dump(jdata, f, sort_keys=True, indent=4, separators=(',', ':'))
			f.close()
	else:
		print('Failed to download {} json: {}'.format(idstr, jres.status))
		print(jres.data)
		return False

	audio_url = 'https://www.freesound.org/apiv2/sounds/{}/download/'.format(info['id'])
	audio_fp = os.path.join(base_dir, idstr + '.' + str(info['type']))
	ares = http.request( 'GET',
						audio_url,
						headers=headers)
	if ares.status == 200:
		with open(audio_fp, 'wb+') as f:
			f.write(ares.data)
			f.close()
	else:
		print('Failed to download {} audio: {}'.format(idstr, ares.status))
		print(ares.data)
		return False

	print('Successfully download {}'.format(idstr))
	return True

def download(res_dict, base_dir):
	from multiprocessing import Pool
	if not os.path.exists(base_dir):
		os.makedirs(base_dir)

	for _id in res_dict:
		if not _download_one((res_dict[_id],base_dir)): break
	# pool = Pool(processes=10)
	# rl = [(r, base_dir) for r in res_list]
	# _r = pool.map(download_one, rl)
	# pool.close()
	# pool.join()



