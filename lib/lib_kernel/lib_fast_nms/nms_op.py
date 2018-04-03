# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'fast_nms.so')
_nms_module = tf.load_op_library(filename)
nms = _nms_module.nms
