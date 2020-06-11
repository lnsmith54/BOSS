import numpy as np
import tensorflow as tf

pLabels = np.array([0, 1, 2, 3, 2, 1, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
print("pLabels ",pLabels)

indx = [np.zeros_like(pLabels,dtype=int)]
for i in range(1,4):
	tmp=i*np.ones_like(pLabels,dtype=int)
	indx = np.concatenate((indx,[tmp]), axis=0)
pLabels = np.tile(pLabels,(4,1))
print("pLabels ",pLabels)
print("indx ",indx)

comparing = tf.math.equal(pLabels,indx)
c=tf.math.reduce_sum(tf.cast(comparing,tf.float32),axis=1)

with tf.Session() as sess:
    counts = sess.run(c)
print(counts)