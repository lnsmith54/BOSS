import tensorflow as tf
import numpy as np
#pseudo_mask = tf.placeholder(tf.float32, shape=(12))

delT = 0.2
confidence = 0.95
pLabels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 4, 1])
#classes, idx, class_count = tf.unique_with_counts(pLabels)
#classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#class_count = np.array([0, 2, 1, 1, 2, 2, 1, 2, 1, 0])
pLabels = tf.cast(pLabels,dtype=tf.int64)
classes, idx, counts = tf.unique_with_counts(pLabels)
#class_count = tf.zeros([10],dtype=tf.int32)
shape = tf.constant([10])
classes = tf.cast(classes,dtype=tf.int32)
class_count = tf.scatter_nd(tf.reshape(classes,[8,1]),counts, shape)

with tf.Session() as sess:
    rs = sess.run(class_count)
    print(rs)

print("class_count ", class_count)
class_count= tf.cast(class_count,dtype=tf.float32)
print("class_count ", class_count)
ratios  = tf.math.divide_no_nan(tf.ones_like(class_count,dtype=tf.float32),class_count)
print("pLabels.shape[-1] ",pLabels.shape[-1])
ratio = tf.gather_nd(ratios, tf.reshape(pLabels,[tf.size(pLabels),1]) )
ratio = tf.math.divide_no_nan(tf.scalar_mul(12.0, ratio), tf.reduce_sum(ratio))
with tf.Session() as sess:
    rs = sess.run(ratio)
    print(rs)

mxCount = tf.reduce_max(class_count, axis=0)
ratios  = 1.0 - tf.math.divide_no_nan(class_count, mxCount)
ratios  = confidence - delT*ratios
confidences =  tf.gather_nd(ratios, tf.reshape(pLabels,[tf.size(pLabels),1]) )

with tf.Session() as sess:
    rs = sess.run(confidences)
    print(rs)

pseudo_mask = ratio >= confidences
with tf.Session() as sess:
    rs = sess.run(pseudo_mask)
    print(rs)

subLabels = tf.boolean_mask(pLabels,pseudo_mask)
with tf.Session() as sess:
    rs = sess.run(subLabels)
    print(rs)

classes, idx, counts = tf.unique_with_counts(pLabels)

#pred = np.array([[1, 5, 9, 2],[10, 1, 7, 4],[2, 8, 1, 3]])
#print(pred)
#print(np.argsort(pred,axis=0))
#print(np.argsort(pred,axis=0).argmax(axis=0))


#                for i in range(self.dataset.nclass):
#                    class_i = classes[i]*tf.ones_like(pLabels,dtype=tf.float32)
#                    class_i = i*tf.ones_like(pLabels,dtype=tf.float32) 
#                    class_i = tf.cast(tf.math.equal(pLabels,class_i),dtype=tf.float32)
#                    confidences = tf.math.add(confidences,tf.scalar_mul( confidence - delT*(1.0-ratios[i]) , class_i) )
