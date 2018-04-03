# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from detection_opr.utils.bbox_transform import bbox_transform


def anchor_target_layer(
        gt_boxes, im_info, _feat_stride, all_anchors, num_anchors,
        is_restrict_bg=False):
    """Same as the anchor target layer in original Fast/er RCNN """
    # A = num_anchors
    # K = total_anchors / num_anchors

    total_anchors = all_anchors.shape[0]
    im_info = im_info[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[
        gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    if is_restrict_bg:
        num_bg = max(num_bg, num_fg * 1.5)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    # labels
    # labels = labels.reshape((1, height, width, A))
    rpn_labels = labels.reshape((-1, 1))

    # bbox_targets
    bbox_targets = bbox_targets.reshape((-1, 4))
    rpn_bbox_targets = bbox_targets

    return rpn_labels, rpn_bbox_targets


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(
        np.float32, copy=False)
    if 'RPN_NORMALIZE_TARGETS' in cfg.TRAIN.keys() and cfg.TRAIN.RPN_NORMALIZE_TARGETS:
        assert cfg.TRAIN.RPN_NORMALIZE_MEANS is not None
        assert cfg.TRAIN.RPN_NORMALIZE_STDS is not None
        targets -= cfg.TRAIN.RPN_NORMALIZE_MEANS
        targets /= cfg.TRAIN.RPN_NORMALIZE_STDS
    return targets
