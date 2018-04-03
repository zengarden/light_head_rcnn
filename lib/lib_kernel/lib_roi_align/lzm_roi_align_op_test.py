import tensorflow as tf
import numpy as np
import roi_align_op
import roi_align_op_grad
import tensorflow as tf
import pdb


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(32, 100, 100, 3)
#array = np.arange(100)
#array = array.reshape(1, 10, 10, 1)
#array = np.ones((32, 100, 100, 3))
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 0, 0, 6, 6]], dtype=tf.float32)

W = weight_variable([3, 3, 3, 1])
h = conv2d(data, W)

[y, argmax] = roi_align_op.roi_align(h, rois, 7, 7, 1, 1, 1.0)
#pdb.set_trace()
y_data = tf.convert_to_tensor(np.ones((1, 7, 7, 1)), dtype=tf.float32)
print(y_data, y, argmax)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
#a, b = sess.run([y, argmax])
#from IPython import embed; embed()
#print(sess.run(y))
#pdb.set_trace()
for step in range(2):
   sess.run(train)
   print(step, sess.run(W))
   #print(sess.run(y))

#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
