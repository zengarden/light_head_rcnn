import tensorflow as tf
import numpy as np
import pdb
import os
import nms_op

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# pdb.set_trace()

rois = tf.convert_to_tensor(
    [[0, 1, 2, 3, 4], [1, 2, 3, 4, 4], [2, 3, 4, 5, 2]],
    dtype=tf.float32)

nms_out = nms_op.nms(rois, 0.1, 3)

hehe = nms_out[0][0:nms_out[1][0]]
# keep_out = nms_out[0][:nms_out[1]]

# hh= tf.transpose(hh, [0, 3, 1, 2])
# [y2, channels, argmax_position] = psalign_pooling_op.psalign_pool(
#     hh, rois, group_size=5, sample_height=2,
#     sample_width=2, spatial_scale=1.0)
# [y2, channels] = psalign_pooling_op.psalign_pool(
#     hh, rois, 5, 2, 2, 1.0)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

print(sess.run(rois))
print("-------")
print(sess.run( nms_out))
