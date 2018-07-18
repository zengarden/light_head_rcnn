# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

from IPython import embed
from config import cfg
from detection_opr.box_utils.bbox_transform_opr import bbox_transform_inv, \
    clip_boxes

import tensorflow as tf
import numpy as np


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = tf.where((ws >= min_size) & (hs >= min_size))[:, 0]
    return keep


def proposal_opr(
        rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors,
        num_anchors, is_tfchannel=False, is_tfnms=False):
    """ Proposal_layer with tensors
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == 'TRAIN':
        pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
        batch = cfg.train_batch_per_gpu
    else:
        pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH
        batch = cfg.test_batch_per_gpu

    if is_tfchannel:
        scores = tf.reshape(rpn_cls_prob, (batch, -1, 2))
        scores = scores[:, :, 1]
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (batch, -1, 4))
    else:
        from IPython import embed
        print("other channel type not implemented")
        embed()

    if 'RPN_NORMALIZE_TARGETS' in cfg.TRAIN.keys() \
            and cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        rpn_bbox_pred *= cfg.TRAIN.RPN_NORMALIZE_STDS
        rpn_bbox_pred += cfg.TRAIN.RPN_NORMALIZE_MEANS

    min_size = 0
    if cfg_key == 'TRAIN':
        if 'RPN_MIN_SIZE' in cfg.TRAIN.keys():
            min_size = cfg.TRAIN.RPN_MIN_SIZE
    elif cfg_key == 'TEST':
        if 'RPN_MIN_SIZE' in cfg.TEST.keys():
            min_size = cfg.TEST.RPN_MIN_SIZE

    batch_scores = []
    batch_proposals = []
    for b_id in range(batch):
        cur_im_info = im_info[b_id]
        cur_scores = scores[b_id]
        cur_rpn_bbox_pred = rpn_bbox_pred[b_id]

        cur_scores = tf.squeeze(tf.reshape(cur_scores, (-1, 1)), axis=1)
        cur_proposals = bbox_transform_inv(anchors, cur_rpn_bbox_pred)
        cur_proposals = clip_boxes(cur_proposals, cur_im_info[:2])

        if min_size > 0:
            assert 'Set MIN_SIZE will make mode slow with tf.where opr'
            keep = filter_boxes(cur_proposals, min_size * cur_im_info[2])
            cur_proposals = tf.gather(cur_proposals, keep, axis=0)
            cur_scores = tf.gather(cur_scores, keep, axis=0)

        if pre_nms_topN > 0:
            cur_order = tf.nn.top_k(cur_scores, pre_nms_topN, sorted=True)[1]
            cur_proposals = tf.gather(cur_proposals, cur_order, axis=0)
            cur_scores = tf.gather(cur_scores, cur_order, axis=0)

        if is_tfnms:
            tf_proposals = cur_proposals + np.array([0, 0, 1, 1])
            keep = tf.image.non_max_suppression(
                tf_proposals, cur_scores, post_nms_topN, nms_thresh)
        else:
            from lib_kernel.lib_fast_nms import nms_op
            keep, keep_num, mask, _ = nms_op.nms(
                cur_proposals, nms_thresh, post_nms_topN)
            keep = keep[:keep_num[0]]

        cur_proposals = tf.gather(cur_proposals, keep, axis=0)
        cur_scores = tf.gather(cur_scores, keep, axis=0)

        batch_inds = tf.ones((tf.shape(cur_proposals)[0], 1)) * b_id
        rois = tf.concat((batch_inds, cur_proposals), axis=1)
        batch_proposals.append(rois)
        batch_scores.append(cur_scores)

    final_proposals = tf.concat(batch_proposals, axis=0)
    final_scores = tf.concat(batch_scores, axis=0)
    return final_proposals, final_scores

def debug_single(x, y):
    from IPython import embed
    embed()
    return True
