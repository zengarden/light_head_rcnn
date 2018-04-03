# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#todo change bbox_transform to oprerator
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def _concat_new_axis(t1, t2, t3, t4, axis):
    return tf.concat(
        [tf.expand_dims(t1, -1), tf.expand_dims(t2, -1),
         tf.expand_dims(t3, -1), tf.expand_dims(t4, -1)], axis=axis)


def bbox_transform_inv(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = tf.expand_dims(boxes[:, 0] + 0.5 * widths, -1)
    ctr_y = tf.expand_dims(boxes[:, 1] + 0.5 * heights, -1)

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    widths = tf.expand_dims(widths, -1)
    heights = tf.expand_dims(heights, -1)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    # x1
    # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    # y1
    # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    # x2
    # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    # y2
    # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = _concat_new_axis(pred_x1, pred_y1, pred_x2, pred_y2, 2)
    pred_boxes = tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1))
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    x1 = tf.maximum(tf.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    y1 = tf.maximum(tf.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    x2 = tf.maximum(tf.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    y2 = tf.maximum(tf.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)

    pred_boxes = _concat_new_axis(x1, y1, x2, y2, 2)
    pred_boxes = tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1))
    return pred_boxes
