import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'roi_align.so')
_roi_align_module = tf.load_op_library(filename)
roi_align = _roi_align_module.roi_align
roi_align_grad = _roi_align_module.roi_align_grad
