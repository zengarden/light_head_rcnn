# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from detection_opr.utils.bbox_transform import bbox_transform_inv, clip_boxes
from detection_opr.utils.nms_wrapper import nms
import tensorflow as tf
from config import cfg


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride,
                   anchors, num_anchors, is_tfchannel=False):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    if cfg_key == 'TRAIN':
        pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
    else:
        pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh = cfg.TEST.RPN_NMS_THRESH

    im_info = im_info[0]
    # from IPython import embed; embed()
    # Get the scores and bounding boxes
    if is_tfchannel:
        scores = rpn_cls_prob.reshape(-1, 2)
        scores = scores[:, 1]
    else:
        scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    # if cfg_key == 'TRAIN' and 'RPN_NORMALIZE_TARGETS' in cfg.TRAIN.keys() \
    if 'RPN_NORMALIZE_TARGETS' in cfg.TRAIN.keys() \
            and cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        rpn_bbox_pred *= cfg.TRAIN.RPN_NORMALIZE_STDS
        rpn_bbox_pred += cfg.TRAIN.RPN_NORMALIZE_MEANS

    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_info[:2])

    # filter boxes
    min_size = 0
    if cfg_key == 'TRAIN':
        if 'RPN_MIN_SIZE' in cfg.TRAIN.keys():
            min_size = cfg.TRAIN.RPN_MIN_SIZE
    elif cfg_key == 'TEST':
        if 'RPN_MIN_SIZE' in cfg.TEST.keys():
            min_size = cfg.TEST.RPN_MIN_SIZE

    if min_size > 0:
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores.flatten()


def proposal_without_nms_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key,
                               feat_stride, anchors, num_anchors,
                               is_tfchannel=False):
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == 'TRAIN':
        pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
    else:
        pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N
    im_info = im_info[0]
    # Get the scores and bounding boxes
    if is_tfchannel:
        scores = rpn_cls_prob.reshape(-1, 2)
        scores = scores[:, 1]
    else:
        scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))

    if 'RPN_NORMALIZE_TARGETS' in cfg.TRAIN.keys() \
            and cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        rpn_bbox_pred *= cfg.TRAIN.RPN_NORMALIZE_STDS
        rpn_bbox_pred += cfg.TRAIN.RPN_NORMALIZE_MEANS

    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_info[:2])

    # filter boxes
    min_size = 0
    if cfg_key == 'TRAIN':
        if 'RPN_MIN_SIZE' in cfg.TRAIN.keys():
            min_size = cfg.TRAIN.RPN_MIN_SIZE
    elif cfg_key == 'TEST':
        if 'RPN_MIN_SIZE' in cfg.TEST.keys():
            min_size = cfg.TEST.RPN_MIN_SIZE
    if min_size > 0:
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order].flatten()

    ##why add one, because tf nms assume x2,y2 does not include border
    proposals_addone = np.array(proposals)
    proposals_addone[:, 2] += 1
    proposals_addone[:, 3] += 1
    return proposals, scores, proposals_addone


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
