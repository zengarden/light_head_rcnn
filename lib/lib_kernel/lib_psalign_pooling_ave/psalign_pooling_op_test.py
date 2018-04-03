import tensorflow as tf
import numpy as np
import psalign_pooling_op
import psalign_pooling_op_grad
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

# pdb.set_trace()

rois = tf.convert_to_tensor([[0, 0, 0, 4, 4], [0, 0, 0, 2, 4], [
                            0, 0, 0, 4, 2]], dtype=tf.float32)
hh = tf.convert_to_tensor(np.random.rand(1, 5, 5, 25*7), dtype=tf.float32)
#hh= tf.transpose(hh, [0, 3, 1, 2])


[y2, channels, argmax_position] = psalign_pooling_op.psalign_pool(hh, rois, group_size=5, sample_height=2, sample_width=2, spatial_scale=1.0)
# [y2, channels] = psalign_pooling_op.psalign_pool(
#     hh, rois, 5, 2, 2, 1.0)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
from IPython import embed

print(sess.run(hh))
print("-------")
print(sess.run(y2))
