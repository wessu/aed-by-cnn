import time
import gc

class A:
    def __init__(self):
        self.x = 1
        self.y = 2
        self.why = 'no reason'

def time_to_append(size, append_list, item_gen):
    t0 = time.time()
    for i in xrange(0, size):
        append_list.append(item_gen())
    return time.time() - t0

def test():
    x = []
    count = 10000
    for i in xrange(0,1000):
        print len(x), time_to_append(count, x, lambda: A())

def test_nogc():
    x = []
    count = 10000
    for i in xrange(0,1000):
        gc.disable()
        print len(x), time_to_append(count, x, lambda: A())
        gc.enable()

# gc.disable()
aa = [1,2,3]
bb = aa
b = gc.get_referrers(bb)
c = gc.get_referents(bb)
print(repr(b))
print(c)
# gc.enable()