import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'nms.so')
_nms_module = tf.load_op_library(filename)
nms = _nms_module.nms
