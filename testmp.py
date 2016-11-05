from multiprocessing import Lock
from multiprocessing import Manager
from multiprocessing import Pool
def testfunc(a):
	print(a)
def test1(num):
	def test2(num):
		pool = Pool(processes=5)
		pool.map(testfunc, range(num, num+20))
		pool.close()
		pool.join()
	test2(num)

import threading
import time

class myThread (threading.Thread):
    def __init__(self, threadID, name, num):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num = num
    def run(self):
        print "Starting " + self.name
        # Get lock to synchronize threads
        # threadLock.acquire()
        print_time(self.name, self.num)
        # Free lock to release next thread
        # threadLock.release()

def print_time(threadName, num):
    time.sleep(2)
    print("{} go {}".format(threadName, num))
    test1(num)

threadLock = threading.Lock()
threads = []

# Create new threads
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 300)

# Start new Threads
thread1.start()
thread2.start()

# Add threads to thread list
threads.append(thread1)
threads.append(thread2)

# Wait for all threads to complete
for t in threads:
    t.join()
print "Exiting Main Thread"