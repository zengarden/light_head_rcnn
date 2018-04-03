# -*- coding: utf-8 -*-
import sys
from config import config
import numpy as np
import cv2
import time
import json

from IPython import embed
from detection_opr.box_utils.box import BoxUtil


def get_hw_by_short_size(im_height, im_width, short_size, max_size):
    im_size_min = np.min([im_height, im_width])
    im_size_max = np.max([im_height, im_width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max

    resized_height, resized_width = int(round(im_height * scale)), int(
        round(im_width * scale))
    return resized_height, resized_width


def resize_img_by_short_and_max_size(
        img, short_size, max_size, *, random_scale_methods=False):
    resized_height, resized_width = get_hw_by_short_size(
        img.shape[0], img.shape[1], short_size, max_size)
    scale = resized_height / (img.shape[0] + 0.0)

    chosen_resize_option = cv2.INTER_LINEAR
    if random_scale_methods:
        resize_options = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST,
                          cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    img = cv2.resize(img, (resized_width, resized_height),
                     interpolation=chosen_resize_option)
    return img, scale


def flip_image_and_boxes(img, boxes=None, *, segs=None):
    h, w, c = img.shape
    flip_img = cv2.flip(img, 1)
    if segs is not None:
        flip_segs = segs[:, :, ::-1]

    if boxes is not None:
        flip_boxes = boxes.copy()
        for i in range(flip_boxes.shape[0]):
            flip_boxes[i, 0] = w - boxes[i, 2] - 1  # x
            flip_boxes[i, 2] = w - boxes[i, 0] - 1  # x1
        if segs is not None:
            return flip_img, flip_boxes, flip_segs
        else:
            return flip_img, flip_boxes
    else:
        if segs is not None:
            return flip_img, flip_segs
        else:
            return flip_img


def pad_image_and_boxes(
        img, height, width, mean_value, boxes=None):
    """ pad or crop the image and boxes"""
    o_h, o_w, _ = img.shape
    margins = np.zeros(2, np.int32)

    assert o_h <= height
    margins[0] = height - o_h
    img = cv2.copyMakeBorder(
        img, 0, margins[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    img[o_h:, :, :] = mean_value

    assert o_w <= width
    margins[1] = width - o_w
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, margins[1], cv2.BORDER_CONSTANT, value=0)
    img[:, o_w:, :] = mean_value

    if boxes is not None:
        return img, boxes
    else:
        return img


def get_data_for_singlegpu(batch_lines):
    batch_records = []
    hw_stat = np.zeros((config.train_batch_per_gpu, 2), np.int32)
    batch_per_gpu = config.train_batch_per_gpu
    short_size = config.image_short_size
    max_size = config.image_max_size

    for i in range(len(batch_lines)):
        raw_line = batch_lines[i]
        record = json.loads(raw_line)
        batch_records.append(record)
        hw_stat[i, :] = record['height'], record['width']

    if config.batch_image_preprocess == 'pad':
        batch_image_height = np.max(hw_stat[:, 0])
        batch_image_width = np.max(hw_stat[:, 1])
    else:
        from IPython import embed;
        print("other type is not implemented")
        embed()

    # from IPython import embed;
    # embed()
    is_batch_ok = True
    filter_box_size = config.batch_filter_box_size
    batch_resized_height, batch_resized_width = get_hw_by_short_size(
        batch_image_height, batch_image_width, short_size, max_size)

    batch_images = np.zeros(
        (batch_per_gpu, batch_resized_height, batch_resized_width, 3),
        dtype=np.float32)
    batch_gts = np.zeros(
        (batch_per_gpu, config.max_boxes_of_image, config.nr_box_dim),
        dtype=np.float32)
    batch_info = np.zeros(
        (batch_per_gpu, config.nr_info_dim), dtype=np.float32)

    for i in range(batch_per_gpu):
        record = batch_records[i]
        # process the images
        image_path = config.train_root_folder + record['fpath']
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        while img is None:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        gtboxes = record['gtboxes']
        gt_boxes = BoxUtil.parse_gt_boxes(gtboxes)
        gt = np.zeros((len(gt_boxes), 5))
        gt_idx = 0
        for j, gb in enumerate(gt_boxes):
            if gb.ign != 1 or not config.filter_gt_ignore_label:
                gt[gt_idx, :] = [gb.x, gb.y, gb.x1, gb.y1,
                                 config.class_names.index(gb.tag)]
                gt_idx += 1
            elif config.train_gt_ignore_label:
                gt[gt_idx, :] = [gb.x, gb.y, gb.x1, gb.y1,
                                 config.anchor_ignore_label]
                gt_idx += 1

        if gt_idx == 0:
            is_batch_ok = False
            break
        gt = gt[:gt_idx, :]

        padded_image, padded_gt = pad_image_and_boxes(
            img, batch_image_height, batch_image_width,
            config.image_mean, gt)
        # filter the images with box_size < config.train_min_box_size
        hs_gt = padded_gt[:, 3] - padded_gt[:, 1] + 1
        ws_gt = padded_gt[:, 2] - padded_gt[:, 0] + 1
        keep = (ws_gt >= filter_box_size) * (hs_gt >= filter_box_size)
        if keep.sum() == 0:
            is_batch_ok = False
            break
        else:
            padded_gt = padded_gt[keep, :]

        original_height, original_width, channels = padded_image.shape
        resized_image, scale = resize_img_by_short_and_max_size(
            padded_image, short_size, max_size)
        padded_gt[:, 0:4] *= scale
        resized_gt = padded_gt

        nr_gtboxes = resized_gt.shape[0]

        if np.random.randint(2) == 1:
            resized_image, resized_gt = flip_image_and_boxes(
                resized_image, resized_gt)

        resized_image = resized_image.astype(np.float32) - config.image_mean
        batch_images[i] = resized_image[:, :, [2, 1, 0]]
        batch_gts[i, :nr_gtboxes] = resized_gt
        batch_info[i, :] = (
            resized_image.shape[0], resized_image.shape[1], scale,
            original_height, original_width, nr_gtboxes)

    return dict(
        data=batch_images, boxes=batch_gts, im_info=batch_info,
        is_valid=is_batch_ok)
    #return dict(
    #    data=batch_images, boxes=resized_gt,
    #    im_info=np.array(
    #        [resized_image.shape[0], resized_image.shape[1], scale,
    #         original_height, original_width, nr_gtboxes]).reshape(1, -1))


def train_dataset(seed=config.seed_dataprovider):
    source = config.train_source
    with open(source) as f:
        files = f.readlines()
    batch_per_gpu = config.train_batch_per_gpu

    nr_files = len(files)
    # shuffle the dataset
    np.random.seed(seed)
    np.random.shuffle(files)
    file_idx = 0
    while file_idx < nr_files:
        # from IPython import embed;
        # embed()
        single_gpu_data = None
        while single_gpu_data is None or single_gpu_data['is_valid'] == False:
            batch_lines = []
            for i in range(batch_per_gpu):
                batch_lines.append(files[file_idx])
                file_idx += 1
                if file_idx >= nr_files:
                    file_idx = 0
                    np.random.shuffle(files)
            single_gpu_data = get_data_for_singlegpu(batch_lines)
        yield single_gpu_data


def val_dataset():
    root = config.eval_root_folder
    source = config.eval_source
    with open(source) as f:
        files = f.readlines()
    total_files = len(files)

    def read_func(line):
        record = json.loads(line)
        image_id = record['ID']
        image_path = root + record['fpath']
        gtboxes = record['gtboxes']
        gt_boxes = BoxUtil.parse_gt_boxes(gtboxes)
        # ground truth
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return dict(data=img, boxes=gt_boxes, image_id=image_id,
                    image_path=image_path)

    return dict(records=files, nr_records=total_files, read_func=read_func)


if __name__ == "__main__":
    tr = train_dataset()
    for i in range(1000):
        dd = next(tr)
        # show_batches(dd)
        #print(dd['im_info'])
        #print(dd['data'])
        # print(dd['boxes'])
        print('----------')
