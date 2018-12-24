# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
from __future__ import division
from __future__ import print_function

from IPython import embed

from config import cfg
import network_desp
import dataset

import tensorflow.contrib.slim as slim
import tensorflow as tf
import sys, os, pprint, time, glob, argparse, logging, setproctitle, numpy as np

from utils.dpflow.data_provider import DataFromList, MultiProcessMapDataZMQ
from utils.dpflow.prefetching_iter import PrefetchingIter
from utils.tf_utils.model_helper import average_gradients, sum_gradients, \
    get_variables_in_checkpoint_file

from tqdm import tqdm
from utils.py_utils import QuickLogger, misc
from utils.py_faster_rcnn_utils.timer import Timer

import pdb 
import warnings
warnings.filterwarnings("ignore")

def snapshot(sess, saver, epoch, step):
    # 这是要一个epoch才保存一次的节奏啦
    filename = 'epoch_{:d}'.format(epoch) + '.ckpt'
    # e.g. output/zuosi/light_head_rcnn.ori_res101.coco
    model_dump_dir = os.path.join(cfg.output_dir, 'model_dump')
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)
    filename = os.path.join(model_dump_dir, filename)
    saver.save(sess, filename, global_step=step)
    print('Wrote snapshot to: {:s}'.format(filename))


def get_data_flow():
    source = cfg.train_source
    with open(source) as f:
        files = f.readlines()

    data = DataFromList(files)

    # 获取数据最重要的还是这个zmq
    dp = MultiProcessMapDataZMQ(
        data, cfg.nr_dataflow, dataset.get_data_for_singlegpu)
    dp.reset_state()
    dataiter = dp.get_data()
    return dataiter


def train(args):
    logger = QuickLogger(log_dir=cfg.output_dir).get_logger()
    logger.info(cfg)
    np.random.seed(cfg.rng_seed)
    num_gpu = len(args.devices.split(','))
    net = network_desp.Network()
    data_iter = get_data_flow() # 数据迭代器
    prefetch_data_layer = PrefetchingIter(data_iter, num_gpu)

    # tf.device()指定tf运行的GPU或CPU设备
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0.),
            trainable=False)

        # log_device_placement=True会打印出执行操作所使用的设备 
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        # 刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        tf.set_random_seed(cfg.rng_seed)

        lr = tf.Variable(cfg.get_lr(0), trainable=False)
        lr_placeholder = tf.placeholder(lr.dtype, shape=lr.get_shape())
        update_lr_op = lr.assign(lr_placeholder)
        # 定义优化器
        opt = tf.train.MomentumOptimizer(lr, cfg.momentum)

        '''data processing'''
        inputs_list = []
        for i in range(num_gpu):
            inputs_list.append(net.get_inputs())
        put_op_list = []
        get_op_list = []
        for i in range(num_gpu):
            with tf.device("/GPU:%s" % i):
                area = tf.contrib.staging.StagingArea(
                    dtypes=[tf.float32 for _ in range(len(inputs_list[0]))])
                put_op_list.append(area.put(inputs_list[i]))
                get_op_list.append(area.get())
        '''
        tf.train.Coordinator()是用来创建一个线程管理器对象,因为tf的session是支持多线程的,
        可以在同一个session中创建多个线程并行执行.Coordinator对象用来管理session中的多线程,
        可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常.
        '''
        coord = tf.train.Coordinator()
        init_all_var = tf.initialize_all_variables()
        sess.run(init_all_var)
        # QueueRunner类用来协调多个工作线程同时将多个张量推入同一个队列中
        queue_runner = tf.train.start_queue_runners(coord=coord, sess=sess)
        '''end of data processing'''

        tower_grads = []
        biases_regularizer = tf.no_regularizer
        biases_ini = tf.constant_initializer(0.0)
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.weight_decay)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i):
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/gpu:0'):
                            with slim.arg_scope(
                                    [slim.conv2d, slim.conv2d_in_plane,
                                     slim.conv2d_transpose,
                                     slim.separable_conv2d,
                                     slim.fully_connected],
                                    weights_regularizer=weights_regularizer,
                                    biases_regularizer=biases_regularizer,
                                    biases_initializer=biases_ini):
                                loss = net.inference('TRAIN', get_op_list[i])
                                loss = loss / num_gpu
                                if i == num_gpu - 1:
                                    regularization_losses = tf.get_collection(
                                        tf.GraphKeys.REGULARIZATION_LOSSES)
                                    loss = loss + tf.add_n(
                                        regularization_losses)

                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        if len(tower_grads) > 1:
            grads = sum_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        final_gvs = []
        with tf.variable_scope('Gradient_Mult'):
            for grad, var in grads:
                scale = 1.
                # if '/biases:' in var.name:
                #    scale *= 2.
                if 'conv_new' in var.name:
                    scale *= 3.
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gvs.append((grad, var))

        # 梯度更新操作
        apply_gradient_op = opt.apply_gradients(
            final_gvs, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # apply_gradient_op = opt.apply_gradients(grads)

        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=100000)
        '''max_to_keep用来设置保存模型的个数,默认为5,即保存最近的5个模型.如果你想
        每训练一代(epoch)就想保存一次模型,则可以将max_to_keep设置为None或者0'''
        saver = tf.train.Saver(max_to_keep=100000)

        variables = tf.global_variables()
        var_keep_dic = get_variables_in_checkpoint_file(cfg.weight)
        var_keep_dic.pop('global_step')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(update_lr_op, {lr_placeholder: cfg.get_lr(0) * num_gpu})
        sess.run(tf.variables_initializer(variables, name='init'))

        variables_to_restore = []
        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                # print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)
        restorer = tf.train.Saver(variables_to_restore)

        restorer.restore(sess, cfg.weight)

        train_collection = net.get_train_collection()
        sess2run = []
        sess2run.append(train_op)
        sess2run.append(put_op_list)
        for col in train_collection.values():
            sess2run.append(col)

        timer = Timer()

        # warm up staging area
        inputs_names = net.get_inputs(mode=1)
        logger.info("start warm up")
        for _ in range(4):
            blobs_list = prefetch_data_layer.forward()
            feed_dict = {}
            for i, inputs in enumerate(inputs_list):
                # blobs = next(data_iter)
                blobs = blobs_list[i]
                for it_idx, it_inputs_name in enumerate(inputs_names):
                    feed_dict[inputs[it_idx]] = blobs[it_inputs_name]
            sess.run([put_op_list], feed_dict=feed_dict)

        logger.info("start train")
        for epoch in range(cfg.max_epoch):
            if epoch == 0 and cfg.warm_iter > 0:
                # pbar = tqdm(range(cfg.warm_iter)) # 预热默认是500次迭代
                pbar = range(cfg.warm_iter) # 预热默认是500次迭代
                up_lr = cfg.get_lr(0) * num_gpu
                bottom_lr = up_lr * cfg.warm_fractor
                iter_delta_lr = 1.0 * (up_lr - bottom_lr) / cfg.warm_iter
                cur_lr = bottom_lr
                for iter in pbar:
                    sess.run(update_lr_op, {lr_placeholder: cur_lr})
                    cur_lr += iter_delta_lr
                    feed_dict = {}
                    blobs_list = prefetch_data_layer.forward()
                    for i, inputs in enumerate(inputs_list):
                        # blobs = next(data_iter)
                        blobs = blobs_list[i]
                        for it_idx, it_inputs_name in enumerate(inputs_names):
                            feed_dict[inputs[it_idx]] = blobs[it_inputs_name]
                    sess_ret = sess.run(sess2run, feed_dict=feed_dict)

                    if iter % cfg.disp_interval == 0:
                        print_str = 'iter %d, ' % (iter)
                        for idx_key, iter_key in enumerate(train_collection.keys()):
                            print_str += iter_key + ': %.4f, ' % sess_ret[
                                idx_key + 2]

                        print_str += 'lr: %.4f, speed: %.3fs/iter' % \
                                     (cur_lr, timer.average_time)
                        logger.info(print_str)
                        # pbar.set_description(print_str)

            '''nr_image_per_epoch这个视数据集不同而不同,像coco2014这个数就是80k的样子,
            每个cpu装入train_batch_per_gpu张图片,总共num_gpu个gpu,以此来计算需要多
            少次迭代, +1是因为range的原因'''
            # pbar = tqdm(range(1, cfg.nr_image_per_epoch // (num_gpu * cfg.train_batch_per_gpu) + 1))
            pbar = range(1, cfg.nr_image_per_epoch // (num_gpu * cfg.train_batch_per_gpu) + 1)
            cur_lr = cfg.get_lr(epoch) * num_gpu
            sess.run(update_lr_op, {lr_placeholder: cur_lr})
            logger.info("epoch: %d" % epoch)
            for iter in pbar:
                timer.tic()
                feed_dict = {}
                blobs_list = prefetch_data_layer.forward()
                for i, inputs in enumerate(inputs_list):
                    # blobs = next(data_iter)
                    blobs = blobs_list[i]
                    for it_idx, it_inputs_name in enumerate(inputs_names):
                        feed_dict[inputs[it_idx]] = blobs[it_inputs_name]

                sess_ret = sess.run(sess2run, feed_dict=feed_dict)
                timer.toc()

                if iter % cfg.disp_interval == 0:
                    print_str = 'iter %d, ' % (iter)
                    for idx_key, iter_key in enumerate(train_collection.keys()):
                        print_str += iter_key + ': %.4f, ' % sess_ret[idx_key + 2]

                    print_str += 'lr: %.4f, speed: %.3fs/iter' % \
                                 (cur_lr, timer.average_time)
                    logger.info(print_str)
                    # pbar.set_description(print_str)

                if iter % cfg.snapshot_interval == 0:
                    snapshot(sess, saver, epoch, global_step)

            snapshot(sess, saver, epoch, global_step)
        coord.request_stop()
        coord.join(queue_runner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test network')
    parser.add_argument(
        '-d', '--devices', default='0', type=str, help='device for training')
    args = parser.parse_args()
    args.devices = misc.parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    setproctitle.setproctitle('train ' + cfg.this_model_dir)
    train(args)
