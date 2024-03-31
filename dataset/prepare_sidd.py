#!/usr/bin/env python
# coding=utf-8
import os
import cv2

import tensorflow as tf

from model import helper
from dataset import tools
from utils import print_log


def prepare_ssid_data(config):

    # 裁剪按照 短边进行计算
    stride = config.img_size

    ds_path = helper.get_record_path(config)
    train_writer = tf.python_io.TFRecordWriter(ds_path)
    eval_path = helper.get_record_path(config, is_eval=True)
    eval_writer = tf.python_io.TFRecordWriter(eval_path)
    # 找出所有噪点的图片
    command = f"find {config.data_dir} -name '*NOISY_*.PNG' -type f"
    file_paths = os.popen(command).readlines()
    train_count = 0
    eval_count = 0
    for idx, img_path in enumerate(sorted(file_paths)):
        img_path = img_path.strip()
        filename = os.path.basename(img_path)

        print_log(f'{img_path} init ...', end=True)

        # 都按照bgr直接读取，裁剪后再转换格式
        # 读取有噪点的图片, 输入
        img_noisy = cv2.imread(img_path)
        # 输出
        img_gt = cv2.imread(img_path.replace('NOISY_', 'GT_'))
        if img_gt.shape[:2] != img_noisy.shape[:2]:
            continue
        h, w = img_gt.shape[:2]  # shape有可能是2维，也可能是3维

        # 把高清图片缩小
        if config.resize:
            # 因为数据集都是横图，所以按照高是resize来处理
            h, w = config.resize, int(w * config.resize / h)
            img_noisy = cv2.resize(img_noisy, (w, h))
            img_gt = cv2.resize(img_gt, (w, h))

        shape = None
        corp_x, corp_y = config.img_size * config.scale, config.img_size * config.scale
        clip_px = 8  # 注意到真实图像中图像边界周围的非边际退化，因此排除了开始和最后的8行/列。
        x_locs = list(range(clip_px, h - corp_y + 1 - clip_px, stride))
        y_locs = list(range(clip_px, w - corp_x + 1 - clip_px, stride))
        for x_idx, x in enumerate(x_locs):  # +1 是为了处理正好h=n*corp_x的时候
            for y_idx, y in enumerate(y_locs):
                if x + corp_x > h or y + corp_y > w:
                    continue
                use_in_eval = False
                # exclude_idx = 3 if config.resize else 6
                # use_in_eval = x_idx < exclude_idx and y_idx < exclude_idx
                # if x_idx > (len(x_locs) - exclude_idx) and y_idx > (len(y_locs) - exclude_idx):
                #     if not use_in_eval:
                #         if config.resize:
                #             # 若是裁剪后的，则用于验证集
                #             use_in_eval = True
                #         else:
                #             # 不用于eval的后面几行列丢弃
                #             continue

                input_image, label_image = tools.crop_sidd_image(img_noisy, img_gt, (corp_x, corp_y), x, y, config)
                shape = input_image.shape[:2] if not shape else shape

                feature = {
                    'input': tools.bytes_feature(input_image.tostring()),
                    'label': tools.bytes_feature(label_image.tostring()),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                if use_in_eval:
                    eval_writer.write(example.SerializeToString())
                    eval_count += 1
                else:
                    train_writer.write(example.SerializeToString())
                    train_count += 1

    train_writer.close()
    # 把个数放到文件名中
    new_path = helper.get_record_path(config, count=train_count)
    os.rename(ds_path, new_path)
    print_log(f'train count is {train_count} {new_path}', end=True)

    eval_writer.close()
    # 把个数放到文件名中
    new_path = helper.get_record_path(config, count=eval_count, is_eval=True)
    os.rename(eval_path, new_path)

    if config.channel_type in [tools.ChannelType.YUV420P.value]:
        # 记录转换后的图片大小
        helper.set_record_info(config, shape)

    print_log(f'eval count is {eval_count} {new_path} , store shape is {shape}', end=True)
    return train_count, eval_count
