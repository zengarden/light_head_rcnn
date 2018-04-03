import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'psalign_pooling.so')
_psalign_pooling_module = tf.load_op_library(filename)
psalign_pool = _psalign_pooling_module.ps_align_pool
psalign_pool_grad = _psalign_pooling_module.ps_align_pool_grad