#!/usr/bin/env python
# coding=utf-8
import time

import numpy as np
import tensorflow as tf

import utils
from model import helper
from dataset import tools
from dataset.prepare_for_test import TestDataSet


def do_test(args):
    # 是否是 验证
    checkpoint_model_path, _ = helper.get_checkpoint_path_and_step(args)

    # 存储位置相关
    save_dir = helper.get_result_path(args)

    utils.print_log(f'checkpoint_model_path is {checkpoint_model_path}, save_dir {save_dir}', end=True)

    dataset = TestDataSet(args)
    args.dataset_count = dataset.count
    count = dataset.count

    model = helper.get_model_func(args)

    avg_time = 0
    for idx, input_, _, filename in dataset:
        tf.reset_default_graph()  # 去掉的话，第二轮就会报错
        with tf.Session() as sess:
            utils.print_log(f'start predict idx={idx + 1}/{count} {filename}', end=True)

            input_ = tools.convert_img(input_, args.channel_type)

            if args.channel_type == tools.ChannelType.YUV.value:
                copy_uv = input_[:, :, 1:]
                if args.channel == 1:
                    input_ = input_[:, :, 0]

            if len(input_.shape) == 2:
                # 二维扩展4维
                input_ = input_[np.newaxis, :, :, np.newaxis]
            else:
                input_ = input_[np.newaxis, :]
            # 归一化
            input_ = input_ / 255.0

            input_images = tf.placeholder(tf.float32, input_.shape, name='images')
            pre_outputs = model(input_images, args)

            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_model_path)

            time_ = time.time()
            result = sess.run(pre_outputs, feed_dict={input_images: input_})

            out_img = tools.recover_to_img(result)
            if args.channel_type == tools.ChannelType.YUV.value:
                # 变成2维
                out_img = np.squeeze(out_img, axis=2)
                # 把uv信息和Y信息合并，然后保存起来
                out_img = np.insert(copy_uv, 0, out_img, axis=2)

            base_name = filename.split('.')[0]
            tools.save_img(out_img, f'{save_dir}/t-{base_name}.jpg', args)

            cost = time.time() - time_
            avg_time += cost

            log = "image: %d/%d, time: %.4f s" % (idx + 1, count, cost)
            utils.print_log(log, end=True)

    utils.print_log('', end=True)
    utils.print_log(f"Avg. Time: {avg_time / count} s", end=True)
