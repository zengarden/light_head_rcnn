# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import tensorflow as tf


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    abs_box_diff = tf.abs(box_diff)
    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, axis=1)
    # rpn_select = tf.where(tf.equal(label, 1))
    rpn_select = tf.where(tf.greater(label, 0))
    value_select = tf.gather(value, rpn_select)
    mask_ig = tf.stop_gradient(
        1.0 - tf.to_float(tf.equal(label, -1)))
    bbox_loss = tf.reduce_sum(value_select) / \
                tf.maximum(1.0, tf.reduce_sum(mask_ig))
    return bbox_loss


def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, nr_classes, sigma=1.0):
    out_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, nr_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, nr_classes, 4])

    value = _smooth_l1_loss_base(
        bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, nr_classes])

    inner_mask = tf.one_hot(
        tf.reshape(label, (-1, 1)), depth=nr_classes, axis=1)
    inner_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inner_mask, [-1, nr_classes])))

    bbox_loss = tf.reduce_sum(tf.reduce_sum(value * inner_mask, 1) * out_mask) \
                / tf.to_float((tf.shape(bbox_pred)[0]))
    return bbox_loss


def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  nr_ohem_sampling, nr_classes, sigma=1.0):
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, labels=label)

    out_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))
    bbox_pred = tf.reshape(bbox_pred, [-1, nr_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, nr_classes, 4])
    value = _smooth_l1_loss_base(
        bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, nr_classes])
    inner_mask = tf.one_hot(tf.reshape(label, (-1, 1)), depth=nr_classes,
                            axis=1)
    inner_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inner_mask, [-1, nr_classes])))
    box_loss = tf.reduce_sum(value * inner_mask, 1) * out_mask

    cls_box_loss = cls_loss + box_loss
    nr_ohem_sampling = tf.minimum(nr_ohem_sampling,
                                  tf.shape(cls_box_loss)[0])

    topk_val, topk_idx = tf.nn.top_k(cls_box_loss, k=nr_ohem_sampling,
                                     sorted=True, name='ohem_loss_index')

    cls_loss_ohem = tf.gather(cls_loss, topk_idx, name='ohem_cls_loss')
    box_loss_ohem = tf.gather(box_loss, topk_idx, name='ohem_box_loss')

    box_loss_ohem = tf.reduce_sum(box_loss_ohem) / tf.to_float(nr_ohem_sampling)
    cls_norm = tf.stop_gradient(tf.minimum(nr_ohem_sampling,
                                           tf.shape(topk_val)[0]))
    cls_loss_ohem = tf.reduce_sum(cls_loss_ohem) / tf.to_float(cls_norm)

    return cls_loss_ohem, box_loss_ohem


def smooth_l1_loss_ohem(bbox_pred, bbox_targets, nr_ohem_sampling, sigma=1.0):
    value = _smooth_l1_loss_base(
        bbox_pred, bbox_targets, sigma=sigma)
    box_loss = tf.reduce_sum(value, axis=1)

    topk_val, topk_idx = tf.nn.top_k(
        box_loss, k=nr_ohem_sampling,
        sorted=False, name='ohem_box_loss_index')
    box_loss_ohem = tf.gather(box_loss, topk_idx, name='ohem_box_loss')
    box_loss_ohem = tf.reduce_sum(box_loss_ohem) / nr_ohem_sampling
    return box_loss_ohem


def softmax_loss_ohem(cls_score, label, nr_ohem_sampling):
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, labels=label)
    topk_val, topk_idx = tf.nn.top_k(cls_loss, k=nr_ohem_sampling,
                                     sorted=False, name='ohem_cls_loss_index')
    cls_loss_ohem = tf.gather(cls_loss, topk_idx, name='ohem_cls_loss')
    cls_loss_ohem = tf.reduce_sum(cls_loss_ohem) / nr_ohem_sampling
    return cls_loss_ohem


def focus_loss(
        prob, label, gamma=2.0, alpha=0.25, is_make_onehot=True, nr_cls=None):
    if is_make_onehot:
        label = tf.one_hot(label, depth=nr_cls)
    pt = tf.reduce_sum(tf.to_float(label) * tf.to_float(prob), axis=1)
    loss = -1.0 * alpha * tf.pow(1 - pt, gamma) * tf.log(pt + 1e-14)
    return loss
