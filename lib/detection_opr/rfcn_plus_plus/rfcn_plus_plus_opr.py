# encoding: utf-8
"""
The MIT License (MIT)

Copyright (c) 2013 Thomas Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

@author: zeming li, yilun chen
@contact: zengarden2009@gmail.com, gentlesky0@gmail.com
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#from lib_kernel.lib_roifm_maxk_mask import roifm_maxk_mask_op
from IPython import embed
def global_context_module(bottom, prefix='', ks=15, chl_mid=256, chl_out=1024):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    col_max = slim.conv2d(bottom, chl_mid, [ks, 1],
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_w_pre' % ks)
    col = slim.conv2d(col_max, chl_out, [1, ks],
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_w' % ks)

    row_max = slim.conv2d(bottom, chl_mid, [1, ks],
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_h_pre' % ks)

    row = slim.conv2d(row_max, chl_out, [ks, 1],
        trainable=True, activation_fn=None,
        weights_initializer=initializer, scope=prefix + '_conv%d_h' % ks)

    s = row + col
    return s

def row_column_max_pooling(bottom, prefix='', window=(7, 7)):
    column_mx = slim.max_pool2d(bottom, [window[0], 1],
        stride=[window[0], 1], scope=prefix + '_column_max')
    row_mx = slim.max_pool2d(bottom, [1, window[1]],
        stride=[1, window[1]], scope=prefix + '_row_max')

    column_mean = slim.avg_pool2d(column_mx, [1, window[1]],
        stride=[1, window[1]], scope=prefix + '_column_mean')
    row_mean = slim.avg_pool2d(row_mx, [window[0], 1],
        stride=[window[0], 1], scope=prefix + '_row_mean')

    return row_mean + column_mean


#def roifm_maxk_mask(bottom, k=[1,2,3,4,3,2,1]):
#    mask1 = roifm_maxk_mask_op.roifm_maxk_mask(
#        tf.transpose(bottom, [0, 3, 1, 2]),
#        k[0],k[1],k[2],k[3],k[4],k[5],k[6])
#    mask1 = tf.transpose(mask1, [0, 2, 3, 1])
#
#    mask2 = roifm_maxk_mask_op.roifm_maxk_mask(
#        tf.transpose(bottom, [0, 3, 2, 1]),
#        k[0],k[1],k[2],k[3],k[4],k[5],k[6])
#    mask2 = tf.transpose(mask2, [0, 3, 2, 1])
#
#    return tf.stop_gradient(mask1 + mask2)
#
#
#def roifm_maxk_mask_layer(bottom, maxk=[1,2,3,4,3,2,1]):
#    batch, height, width, chl = bottom.shape
#    mask = np.zeros(bottom.shape,dtype=np.float32)
#    for b in range(batch):
#        for c in range(chl):
#            for h in range(height):
#                idx = np.argpartition(bottom[b,h,:,c], -maxk[h])[-maxk[h]:]
#                mask[b, h, idx, c] = 1.0
#            for w in range(width):
#                idx = np.argpartition(bottom[b,:,w,c], -maxk[w])[-maxk[w]:]
#                mask[b, idx, w, c] = 1.0
#    return mask
