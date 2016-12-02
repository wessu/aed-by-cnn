import os
import multiprocessing
from multiprocessing import Process
# import theano

def proc_run(m, i, f):
    print('Parent m: ', m)
    print('Parent i: ', i)
    print('Parent f: ', f)
    import theano
    print('theano imported')
    x = theano.tensor.dscalar()
    f = theano.function([x], 2*x)
    print(f(4))

m, i, f = '123', 'yoyoyo', '[1,2,3,4]'

for j in range(2):
    print('start')
    proc_run(m, i, f)


# proc = Process(target=proc_run, args=(m, i, f))
# proc.start()
# print('start child proc')
# proc.join()
# print('child joined')