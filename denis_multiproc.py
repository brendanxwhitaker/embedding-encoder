import multiprocessing as mp
import tensorflow as tf

def _threadedFunc(x, out_q):
   sess = tf.Session()
   result = sess.run(tf.add(x, 3))
   out_q.put(result)

q = mp.Queue()
t1 = mp.Process(target=_threadedFunc, args=(1, q))
t2 = mp.Process(target=_threadedFunc, args=(2, q))
t1.start()
t2.start()
t1.join()
t2.join()

# will print 4 and 5 in some order
print(q.det())
print(q.get())
