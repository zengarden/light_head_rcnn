# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, getpass
import os.path as osp
import numpy as np
import argparse
import sys

from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# ------------ please config ROOT_dir and user when u first using -------------#
root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

lib_path = osp.join(root_dir, 'lib')
add_path(osp.join(root_dir, 'tools'))
add_path(lib_path)


class Config:
    user = getpass.getuser()
    # ---------- generate some dirs, e.g. dump dir, weights dir -------------------#
    output_dir = osp.join(
        root_dir, 'output', user,
        os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]) 
    this_model_dir = osp.split(os.path.realpath(__file__))[0]

    ckpt_dir = osp.join(output_dir, 'model_dump')
    eval_dir = osp.join(output_dir, 'eval_dump')
    tb_dir = osp.join(output_dir, 'tfboard_dump')
    weight = osp.join(root_dir, 'data/imagenet_weights/res101.ckpt')

    program_name = user + ":" + os.path.split(
        os.path.split(os.path.realpath(__file__))[0])[1]

    # ------------------- Data configuration --------------------------------------#

    from datasets_odgt.adas import ADAS as datadb

    # 什么意思?
    image_mean = np.array([102.9801, 115.9465, 122.7717])
    # C.image_mean = np.array([122.7717, 102.9801, 115.9465])
    seed_dataprovider = 3
    nr_dataflow = 2 # 数据预处理进程个数,默认16

    datadb = datadb
    class_num = datadb.num_classes
    num_classes = datadb.num_classes
    class_names = datadb.class_names
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))

    batch_image_preprocess = 'pad'
    train_root_folder = os.path.join(root_dir, 'data/ADAS')
    train_source = os.path.join(
        root_dir, 'data', 'ADAS/odformat/train2017.odgt')

    eval_root_folder = os.path.join(root_dir, 'data/ADAS')
    eval_source = os.path.join(
        root_dir, 'data', 'ADAS/odformat/val2017.odgt')
    eval_json = os.path.join(
        root_dir, 'data', 'ADAS/instances_val2017.json')

    filter_gt_ignore_label = True
    train_gt_ignore_label = False

    '''
    image_short_size = 800
    image_max_size = 1333
    '''

    image_short_size = 600 
    image_max_size = 1000

    '''
    image_short_size = 224
    image_max_size = 373
    '''
    eval_resize = True

    eval_image_short_size = 600
    eval_image_max_size = 1000

    #eval_image_short_size = 800
    #eval_image_max_size = 1333
    #eval_image_short_size = 224
    #eval_image_max_size = 224

    test_max_boxes_per_image = 100
    test_cls_threshold = 0.00
    test_vis_threshold = 0.5
    test_nms = 0.3
    test_save_type = 'adas'

    batch_filter_box_size = 0
    max_boxes_of_image = 100
    nr_box_dim = 5
    nr_info_dim = 6

    stride = [16]
    anchor_scales = [2, 4, 8, 16, 32]
    anchor_ratios = [0.5, 1, 2]
    simga_rpn = 3

    rng_seed = 3
    EPS = 1e-14

    # ------------------------------------ TRAIN config -----------------------#
    train_batch_per_gpu = 1 # 默认是2,考虑到内存消耗改为1
    test_batch_per_gpu = 1
    bn_training = False
    tb_dump_interval = 500
    nr_image_per_epoch = 80000  # detectron 1x setting
    # 为啥设置个这么奇怪的初始学习率?按照train_batch_per_gpu为2算的话
    # basic_lr大概就是0.001的样子
    basic_lr = 5e-4 * train_batch_per_gpu * 1.25 # 5*0.0001*2*1.25=0.00125
    momentum = 0.9
    weight_decay = 0.0001

    from utils.tf_utils import lr_policy
    max_epoch = 30
    warm_iter = 500 #预热,默认500次
    warm_fractor = 1.0 / 3.0
    '''
    multi_stage_lr_policy = lr_policy.MultiStageLR(
        [[19, basic_lr], [25, basic_lr * 0.1], [30, basic_lr * 0.01]])
    '''
    multi_stage_lr_policy = lr_policy.MultiStageLR(
        [[5, basic_lr], [10, basic_lr * 0.1], [15, basic_lr * 0.01]])

    def get_lr(self, epoch):
        return self.multi_stage_lr_policy.get_lr(epoch)

    # add by zuosi
    disp_interval = 1000 #控制显示步长
    snapshot_interval = 2000 #快照步长
    # -----------------------------traditional rcnn config --------------------#
    TRAIN = edict()

    TRAIN.HAS_RPN = True
    TRAIN.DOUBLE_BIAS = False
    TRAIN.BIAS_DECAY = False
    TRAIN.USE_GT = True
    TRAIN.TRUNCATED = False
    TRAIN.ASPECT_GROUPING = True

    TRAIN.FG_FRACTION = 0.25
    TRAIN.FG_THRESH = 0.5
    TRAIN.BG_THRESH_HI = 0.5
    TRAIN.BG_THRESH_LO = 0.0
    TRAIN.BBOX_REG = True
    TRAIN.BBOX_THRESH = 0.5

    # 什么意思?为啥是-1?
    TRAIN.BATCH_SIZE = -1  # rcnn batch size
    TRAIN.nr_ohem_sampling = 256 * train_batch_per_gpu

    TRAIN.BBOX_NORMALIZE_TARGETS = True
    # Deprecated (inside weights)
    TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

    TRAIN.RPN_NORMALIZE_TARGETS = False
    TRAIN.RPN_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    TRAIN.RPN_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

    # IOU >= thresh: positive example
    TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
    TRAIN.RPN_CLOBBER_POSITIVES = False
    TRAIN.RPN_FG_FRACTION = 0.5
    TRAIN.RPN_BATCHSIZE = 256
    TRAIN.RPN_NMS_THRESH = 0.7
    # __C.TRAIN.RPN_MIN_SIZE = 16
    TRAIN.USE_ALL_GT = True
    TRAIN.RPN_PRE_NMS_TOP_N = 12000 # 图片小的时候这个有点多,why?
    #TRAIN.RPN_PRE_NMS_TOP_N = 3000 # 如果训练的时候将图片resize到这个尺寸,要调整NMS前roi的数量
    TRAIN.RPN_POST_NMS_TOP_N = 2000 # nms之后保留的roi,将用于训练

    TEST = edict()
    TEST.BBOX_REG = True
    TEST.HAS_RPN = True
    TEST.RPN_NMS_THRESH = 0.7
    # TEST.RPN_PRE_NMS_TOP_N = 6000
    TEST.RPN_PRE_NMS_TOP_N = 1000 # 224测试集小点吧
    TEST.RPN_POST_NMS_TOP_N = 300 # nms之后保留的roi,将用于测试


config = Config()
cfg = config


def link_log_dir():
    if not os.path.exists(osp.join(config.this_model_dir, 'log')):
        cmd = "ln -s " + config.output_dir + " log"
        os.system(cmd)


def link_tools_dir():
    if not os.path.exists(osp.join(config.this_model_dir, 'tools')):
        cmd = "ln -s " + os.path.join(root_dir, 'tools') + " tools"
        os.system(cmd)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-log', '--linklog', default=False, action='store_true')
    parser.add_argument(
        '-tool', '--link_tools', default=False, action='store_true')

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
    if args.link_tools:
        link_tools_dir()
