# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import numpy as np
from IPython import embed

import megbrain as mgb
import megskull as mgsk
from megskull.opr.all import Concat, CondTake, ZerosLike, Log, Exp, Max, Min
from megskull.cblk.elemwise import safelog


def _concat_new_axis(t1, t2, t3, t4, axis=1):
    return Concat(
        [t1.add_axis(-1), t2.add_axis(-1), t3.add_axis(-1), t4.add_axis(-1)],
        axis=axis)


def _box_ltrb_to_cs_opr(bbox, addaxis=None):
    """ transform the left-top right-bottom encoding bounding boxes
    to center and size encodings"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
    if addaxis is None:
        return bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y
    else:
        return bbox_width.add_axis(addaxis), bbox_height.add_axis(
            addaxis), bbox_ctr_x.add_axis(addaxis), bbox_ctr_y.add_axis(addaxis)


def clip_boxes_opr(boxes, im_info):
    """ Clip the boxes into the image region."""
    # x1 >=0
    box_x1 = Max(Min(boxes[:, 0::4], im_info[1] - 1), 0)
    # y1 >=0
    box_y1 = Max(Min(boxes[:, 1::4], im_info[0] - 1), 0)
    # x2 < im_info[1]
    box_x2 = Max(Min(boxes[:, 2::4], im_info[1] - 1), 0)
    # y2 < im_info[0]
    box_y2 = Max(Min(boxes[:, 3::4], im_info[0] - 1), 0)

    # clip_box = Concat([box_x1, box_y1, box_x2, box_y2], axis=1)
    clip_box = _concat_new_axis(box_x1, box_y1, box_x2, box_y2, 2)\
        .reshape(boxes.shape[0], -1)

    return clip_box

def filter_boxes_opr(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    # keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    keep = (ws >= min_size) * (hs >= min_size)
    # NOTE: In FPN, I have met np.all(keep) = 0(don't know why),
    # thus I add the following line to avoid crash
    # keep = keep + (keep.sum().eq(0))

    keep_index = CondTake(keep, keep, 'EQ', 1).outputs[1]
    return keep_index


def filter_anchors_opr(
        all_anchors, im_height, im_width, allowed_border_height,
        allowed_border_width=None):
    if allowed_border_width is None:
        allowed_border_width = allowed_border_height
    inds_inside = (all_anchors[:, 0] >= -allowed_border_width) * \
                  (all_anchors[:, 1] >= -allowed_border_height) * \
                  (all_anchors[:, 2] < im_width + allowed_border_width) * \
                  (all_anchors[:, 3] < im_height + allowed_border_height)

    inds_inside = CondTake(inds_inside, inds_inside, 'EQ', 1).outputs[1]
    return inds_inside


def bbox_transform_opr(bbox, gt):
    """ Transform the bounding box and ground truth to the loss targets.
    The 4 box coordinates are in axis 1"""

    bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y = _box_ltrb_to_cs_opr(bbox)
    gt_width, gt_height, gt_ctr_x, gt_ctr_y = _box_ltrb_to_cs_opr(gt)

    target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
    target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
    target_dw = Log(gt_width / bbox_width)
    target_dh = Log(gt_height / bbox_height)
    # target = Concat([target_dx, target_dy, target_dw, target_dh], axis=1)
    target = _concat_new_axis(target_dx, target_dy, target_dw, target_dh)
    return target


def bbox_transform_inv_opr(anchors, deltas):
    """ Transforms the learned deltas to the final bbox coordinates, the axis is 1"""
    anchor_width, anchor_height, anchor_ctr_x, anchor_ctr_y = \
        _box_ltrb_to_cs_opr(anchors, 1)
    pred_ctr_x = anchor_ctr_x + deltas[:, 0::4] * anchor_width
    pred_ctr_y = anchor_ctr_y + deltas[:, 1::4] * anchor_height
    pred_width = anchor_width * Exp(deltas[:, 2::4])
    pred_height = anchor_height * Exp(deltas[:, 3::4])

    pred_x1 = pred_ctr_x - 0.5 * pred_width
    pred_y1 = pred_ctr_y - 0.5 * pred_height
    pred_x2 = pred_ctr_x + 0.5 * pred_width
    pred_y2 = pred_ctr_y + 0.5 * pred_height

    pred_box = _concat_new_axis(pred_x1, pred_y1, pred_x2, pred_y2, 2)
    pred_box = pred_box.reshape(pred_box.shape[0], -1)

    return pred_box


def box_overlap_opr(box, gt):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) Tensor
    gt : (K, 4) Tensor
    return: (N, K) Tensor, stores Max(0, intersection/union)
    """
    N = box.shape[0]
    K = gt.shape[0]
    target_shape = (N, K, 4)
    b_box = box.add_axis(1).broadcast(target_shape)
    b_gt = gt.add_axis(0).broadcast(target_shape)

    iw = (
        Min(b_box[:, :, 2], b_gt[:, :, 2]) - \
        Max(b_box[:, :, 0], b_gt[:, :, 0]) + 1)
    ih = (
        Min(b_box[:, :, 3], b_gt[:, :, 3]) - \
        Max(b_box[:, :, 1], b_gt[:, :, 1]) + 1)
    inter = Max(iw, 0) * Max(ih, 0)

    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    area_target_shape = (N, K)
    b_area_box = area_box.add_axis(1).broadcast(area_target_shape)
    b_area_gt = area_gt.add_axis(0).broadcast(area_target_shape)

    union = b_area_box + b_area_gt - inter

    overlaps = Max(inter / union, 0)
    return overlaps


def box_overlap_ignore_opr(box, gt, *, ignore_label=-1, return_separate=False):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) Tensor
    gt : (K, 5) Tensor, the last col shows the labels of gt
    return: (N, K) Tensor, stores Max(0, intersection/union)

    Here, we consider the ignore_label of gt boxes. When compute
    box vs ignored_gt, the overlap is replaced by inter / box_area.
    This operation will force the boxes near to ignore gt_boxes to
    be matched to ignored boxes rather than fg or bg labels.
    """
    N = box.shape[0]
    K = gt.shape[0]
    target_shape = (N, K, 4)
    b_box = box.add_axis(1).broadcast(target_shape)
    b_gt = gt[:, :4].add_axis(0).broadcast(target_shape)

    # intersection of boxes
    iw = (Min(b_box[:, :, 2], b_gt[:, :, 2]) - \
          Max(b_box[:, :, 0], b_gt[:, :, 0]) + 1)
    ih = (Min(b_box[:, :, 3], b_gt[:, :, 3]) - \
          Max(b_box[:, :, 1], b_gt[:, :, 1]) + 1)
    inter = Max(iw, 0) * Max(ih, 0)

    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    area_target_shape = (N, K)
    b_area_box = area_box.add_axis(1).broadcast(area_target_shape)
    b_area_gt = area_gt.add_axis(0).broadcast(area_target_shape)

    union = b_area_box + b_area_gt - inter

    overlaps_normal = Max(inter / union, 0)
    overlaps_ignore = Max(inter / b_area_box, 0)

    gt_ignore_mask = gt[:, 4].eq(ignore_label).add_axis(0).broadcast(
        area_target_shape)

    overlaps_normal *= (1 - gt_ignore_mask)
    overlaps_ignore *= gt_ignore_mask

    if return_separate:
        return overlaps_normal, overlaps_ignore
    else:
        overlaps = overlaps_normal + overlaps_ignore
        return overlaps
