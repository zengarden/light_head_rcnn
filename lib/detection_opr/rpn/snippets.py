# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
# import sys
# sys.path.insert(0, '/unsullied/sharefs/lizeming/lzm_home_large/tf-faster-rfcn-multigpu/lib/')

import numpy as np
from detection_opr.rpn.generate_anchors import generate_anchors
import tensorflow as tf


def generate_anchors_pre(
        height, width, feat_stride, anchor_scales=(8, 16, 32),
        anchor_ratios=(0.5, 1, 2), base_size=16):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(
        ratios=np.array(anchor_ratios), scales=np.array(anchor_scales),
        base_size=base_size)
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
         shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length


def generate_anchors_opr(
        height, width, feat_stride, anchor_scales=(8, 16, 32),
        anchor_ratios=(0.5, 1, 2), base_size=16):
    anchors = generate_anchors(
        ratios=np.array(anchor_ratios), scales=np.array(anchor_scales),
        base_size=base_size)
    shift_x = tf.range(width, dtype=np.float32) * feat_stride
    shift_y = tf.range(height, dtype=np.float32) * feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shifts = tf.stack(
        (tf.reshape(shift_x, (-1, 1)), tf.reshape(shift_y, (-1, 1)),
         tf.reshape(shift_x, (-1, 1)), tf.reshape(shift_y, (-1, 1))))
    shifts = tf.transpose(shifts, [1, 0, 2])
    final_anc = tf.constant(anchors.reshape((1, -1, 4)), dtype=np.float32) + \
          tf.transpose(tf.reshape(shifts, (1, -1, 4)), (1, 0, 2))
    return tf.reshape(final_anc, (-1, 4))


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    hehe = generate_anchors_pre(150, 200, 1.0 / 16)
    h, w = tf.constant(150, dtype=np.float32), tf.constant(200, dtype=np.float32)
    haha = generate_anchors_opr(h, w, 1.0 / 16)
    sess = tf.Session()
    xixi = sess.run(haha)
    print(hehe[0] - xixi)
    embed()
