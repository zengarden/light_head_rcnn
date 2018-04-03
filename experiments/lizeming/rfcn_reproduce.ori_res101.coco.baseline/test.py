# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import embed
from config import cfg, config

import argparse
import dataset
import os.path as osp
import network_desp
import tensorflow as tf
import numpy as np
import cv2, os, sys, math, json, pickle
import time

from tqdm import tqdm
from utils.py_faster_rcnn_utils.cython_nms import nms, nms_new
from utils.py_utils import misc

from multiprocessing import Queue, Process
from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
from functools import partial


def load_model(model_file, dev):
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    net = network_desp.Network()
    inputs = net.get_inputs()
    net.inference('TEST', inputs)
    test_collect_dict = net.get_test_collection()
    test_collect = [it for it in test_collect_dict.values()]
    saver = tf.train.Saver()

    saver.restore(sess, model_file)
    return partial(sess.run, test_collect), inputs


def inference(val_func, inputs, data_dict):
    image = data_dict['data']
    ori_shape = image.shape

    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)
    height, width = resized_img.shape[0:2]

    resized_img = resized_img.astype(np.float32) - config.image_mean
    resized_img = np.ascontiguousarray(resized_img[:, :, [2, 1, 0]])

    im_info = np.array(
        [[height, width, scale, ori_shape[0], ori_shape[1], 0]],
        dtype=np.float32)

    feed_dict = {inputs[0]: resized_img[None, :, :, :], inputs[1]: im_info}

    #st = time.time()
    _, scores, pred_boxes, rois = val_func(feed_dict=feed_dict)
    #ed = time.time()
    #print(ed -st)

    boxes = rois[:, 1:5] / scale

    if cfg.TEST.BBOX_REG:
        pred_boxes = bbox_transform_inv(boxes, pred_boxes)
        pred_boxes = clip_boxes(pred_boxes, ori_shape)

    pred_boxes = pred_boxes.reshape(-1, config.num_classes, 4)
    result_boxes = []
    for j in range(1, config.num_classes):
        inds = np.where(scores[:, j] > config.test_cls_threshold)[0]
        cls_scores = scores[inds, j]
        cls_bboxes = pred_boxes[inds, j, :]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(cls_dets, config.test_nms)
        cls_dets = np.array(cls_dets[keep, :], dtype=np.float, copy=False)
        for i in range(cls_dets.shape[0]):
            db = cls_dets[i, :]
            dbox = DetBox(
                db[0], db[1], db[2] - db[0], db[3] - db[1],
                tag=config.class_names[j], score=db[-1])
            result_boxes.append(dbox)
    if len(result_boxes) > config.test_max_boxes_per_image:
        result_boxes = sorted(
            result_boxes, reverse=True, key=lambda t_res: t_res.score) \
            [:config.test_max_boxes_per_image]

    result_dict = data_dict.copy()
    result_dict['result_boxes'] = result_boxes
    return result_dict


def worker(model_file, dev, records, read_func, result_queue):
    func, inputs = load_model(model_file, dev)
    for record in records:
        data_dict = read_func(record)
        result_dict = inference(func, inputs, data_dict)
        result_queue.put_nowait(result_dict)


def eval_all(args):
    devs = args.devices.split(',')
    misc.ensure_dir(config.eval_dir)
    eval_file = open(os.path.join(config.eval_dir, 'results.txt'), 'a')
    dataset_dict = dataset.val_dataset()
    records = dataset_dict['records']
    nr_records = len(records)
    read_func = dataset_dict['read_func']

    nr_devs = len(devs)
    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        model_file = osp.join(
            config.output_dir, 'model_dump',
            'epoch_{:d}'.format(epoch_num) + '.ckpt')

        pbar = tqdm(total=nr_records)
        all_results = []
        if nr_devs == 1:
            func, inputs = load_model(model_file, devs[0])
            for record in records:
                data_dict = read_func(record)
                result_dict = inference(func, inputs, data_dict)
                all_results.append(result_dict)

                if args.show_image:
                    image = result_dict['data']
                    for db in result_dict['result_boxes']:
                        if db.score > config.test_vis_threshold:
                            db.draw(image)
                    if 'boxes' in result_dict.keys():
                        for db in result_dict['boxes']:
                            db.draw(image)
                    cv2.imwrite('/tmp/hehe.png', image)
                    # cv2.imshow('image', image)
                    # cv2.waitKey(0)
                pbar.update(1)
        else:
            nr_image = math.ceil(nr_records / nr_devs)
            result_queue = Queue(500)
            procs = []
            for i in range(nr_devs):
                start = i * nr_image
                end = min(start + nr_image, nr_records)
                split_records = records[start:end]
                proc = Process(target=worker, args=(
                    model_file, devs[i], split_records, read_func,
                    result_queue))
                print('process:%d, start:%d, end:%d' % (i, start, end))
                proc.start()
                procs.append(proc)
            for i in range(nr_records):
                t = result_queue.get()
                all_results.append(t)
                pbar.update(1)

            for p in procs:
                p.join()

        save_filename = save_result(all_results, config.eval_dir, model_file)
        print('Save to %s finished, start evaulation!' % save_filename)
        saved_stdout = sys.stdout
        sys.stdout = eval_file
        print("\nevaluation epoch {}".format(epoch_num))

        if config.test_save_type == 'coco':
            from datasets_odgt.cocoval import cocoval
            cocoval(save_filename)
        else:
            print("not implement")
            embed()
        sys.stdout = saved_stdout
        eval_file.flush()

    eval_file.close()
    print("\n")


def save_result(all_results, save_path, model_name):
    prefix = ''
    if model_name is not None:
        prefix = os.path.basename(
            os.path.basename(model_name).split('.')[0])

    save_filename = os.path.join(
        save_path, prefix + '.' + config.test_save_type)
    save_file = open(save_filename, 'w')
    coco_records = []
    print('The result will save in file: ' + save_filename)
    for result in tqdm(all_results):
        result_boxes = result['result_boxes']
        if config.test_save_type == 'coco':
            image_filename = result['image_id']
            image_id = int(image_filename.split('.')[0].split('_')[-1])
            for rb in result_boxes:
                record = {}
                record['image_id'] = image_id
                record['category_id'] = config.datadb.classes_originID[rb.tag]
                record['score'] = rb.score
                record['bbox'] = [rb.x, rb.y, rb.w, rb.h]
                coco_records.append(record)
        else:
            raise Exception(
                "Unimplemented save type: " + str(config.test_save_type))
    if config.test_save_type == 'coco':
        save_file.write(json.dumps(coco_records))

    save_file.close()
    return save_filename


def make_parser():
    parser = argparse.ArgumentParser('test network')
    parser.add_argument(
        '-d', '--devices', default='0', type=str, help='device for testing')
    parser.add_argument(
        '--show_image', '-s', default=False, action='store_true')
    parser.add_argument('--start_epoch', '-se', default=1, type=int)
    parser.add_argument('--end_epoch', '-ee', default=-1, type=int)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.devices = misc.parse_devices(args.devices)
    if args.end_epoch == -1:
        args.end_epoch = args.start_epoch

    eval_all(args)
