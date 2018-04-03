# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import tensorflow as tf


def _smooth_l1_loss_base(bbox_pred, bbox_targets, bbox_inside_weights,
                         bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    return out_loss_box


def softmax_layer(bottom, name):
    if name == 'rpn_cls_prob_reshape':
        input_shape = tf.shape(bottom)
        bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)


def softmax_loss_ohem(cls_score, label, nr_ohem_sampling):
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, labels=label)
    topk_val, topk_idx = tf.nn.top_k(cls_loss, k=nr_ohem_sampling,
                                     sorted=False, name='ohem_cls_loss_index')
    cls_loss_ohem = tf.gather(cls_loss, topk_idx, name='ohem_cls_loss')
    cls_loss_ohem = tf.reduce_sum(cls_loss_ohem) / nr_ohem_sampling
    return cls_loss_ohem


# rpn do not direct div norm
# 算loss 不算背景loss

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights,
                   bbox_outside_weights, sigma=1.0, dim=[1]):
    value = _smooth_l1_loss_base(
        bbox_pred, bbox_targets, bbox_inside_weights,
        bbox_outside_weights, sigma=sigma, dim=[1])
    loss = tf.reduce_mean(tf.reduce_sum(value, axis=dim))
    return loss


def smooth_l1_loss_ohem(bbox_pred, bbox_targets, bbox_inside_weights,
                        bbox_outside_weights, nr_ohem_sampling,
                        sigma=1.0, dim=[1]):
    box_loss_base = _smooth_l1_loss_base(
        bbox_pred, bbox_targets, bbox_inside_weights,
        bbox_outside_weights, sigma=sigma, dim=[1])
    box_loss = tf.reduce_sum(box_loss_base, axis=dim)

    topk_val, topk_idx = tf.nn.top_k(
        box_loss, k=nr_ohem_sampling,
        sorted=False, name='ohem_box_loss_index')

    box_loss_ohem = tf.gather(box_loss, topk_idx, name='ohem_box_loss')
    box_loss_ohem = tf.reduce_sum(box_loss_ohem) / nr_ohem_sampling
    return box_loss_ohem


def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  bbox_inside_weights, bbox_outside_weights,
                  nr_ohem_sampling, sigma=1.0, dim=[1]):
    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=cls_score, labels=label)
    box_loss_base = _smooth_l1_loss_base(bbox_pred, bbox_targets,
                                         bbox_inside_weights,
                                         bbox_outside_weights,
                                         sigma=sigma, dim=[1])

    box_loss = tf.reduce_sum(box_loss_base, axis=dim)
    cls_box_loss = cls_loss + box_loss

    nr_ohem_sampling = tf.minimum(nr_ohem_sampling,
                                  tf.shape(cls_box_loss)[0])

    topk_val, topk_idx = tf.nn.top_k(cls_box_loss, k=nr_ohem_sampling,
                                     sorted=True, name='ohem_loss_index')

    cls_loss_ohem = tf.gather(cls_loss, topk_idx, name='ohem_cls_loss')
    box_loss_ohem = tf.gather(box_loss, topk_idx, name='ohem_box_loss')

    box_loss_ohem = tf.reduce_sum(box_loss_ohem) / \
                    tf.to_float(nr_ohem_sampling)
    cls_norm = tf.stop_gradient(tf.minimum(nr_ohem_sampling,
                                           tf.shape(topk_val)[0]))

    # db_cls_norm = tf.py_func(debug_single, [cls_loss, box_loss, topk_idx, 
    # cls_loss_ohem, box_loss_ohem, cls_norm], [tf.bool])
    # with tf.control_dependencies(db_cls_norm):
    cls_loss_ohem = tf.reduce_sum(cls_loss_ohem) / tf.to_float(cls_norm)

    return cls_loss_ohem, box_loss_ohem


'''following are not tested'''


def debug_single(x):
    from IPython import embed
    embed()
    return True


def debug_two(x, y):
    from IPython import embed
    embed()
    return True


def debug_four(cls_loss, box_loss, topk_idx,
               cls_loss_ohem, box_loss_ohem, cls_norm):
    from IPython import embed
    embed()
    return True


def _get_mask_of_label(label, background, ignore_label):
    mask_fg = 1 - tf.to_float(tf.equal(label, background))
    mask_ig = 1 - tf.to_float(tf.equal(label, ignore_label))
    # mask_fg = 1 - label.eq(background)
    # mask_ig = 1 - label.eq(ignore_label)
    mask = mask_fg * mask_ig
    return mask, mask_ig


def smooth_l1_loss_valid(bbox_pred, bbox_targets, bbox_inside_weights,
                         bbox_outside_weights, label,
                         background=0, ignore_label=-1,
                         sigma=1.0, dim=[1]):
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, bbox_inside_weights,
                                 bbox_outside_weights, sigma, dim=[1])
    mask, mask_ig = _get_mask_of_label(label, background, ignore_label)
    norm = tf.maximum(1, tf.reduce_sum(mask_ig))
    loss = tf.reduce_sum(value, dim) / norm
    return loss
