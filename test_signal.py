import sys 
import os

if __name__ == '__main__':
	m = sys.argv[1]
	i = sys.argv[2]
	f = sys.argv[3]
	print('Child m:', m)
	print('Child i:', i)
	print('Child f:', f)
	import theano


