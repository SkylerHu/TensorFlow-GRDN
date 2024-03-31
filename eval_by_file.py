#!/usr/bin/env python
# coding=utf-8
import time
import shutil

import cv2
import numpy as np
import tensorflow as tf

import utils
from model import helper
from dataset import tools
from dataset.sidd_for_eval import ssid_eval_data


def do_eval(args):
    # 存储位置相关
    save_dir = helper.get_result_path(args) if args.save_eval_dir else None
    checkpoint_model_path, _ = helper.get_checkpoint_path_and_step(args)
    utils.print_log(f'checkpoint_model_path is {checkpoint_model_path}, save_dir {save_dir}', end=True)

    dataset = ssid_eval_data(args.data_dir, config=args)
    count = 0

    model = helper.get_model_func(args)

    compare = tools.CompareInfo()
    compare2 = tools.CompareInfo()
    is_single_y = args.channel_type == tools.ChannelType.YUV.value and args.channel == 1
    is_use_420p = args.channel_type == tools.ChannelType.YUV420P.value
    eval_size = args.eval_size
    for idx, input_, label_ori, filename in dataset:
        if args.local and idx > 1:
            break
        tf.reset_default_graph()  # 去掉的话，第二轮就会报错
        with tf.Session() as sess:
            utils.print_log(f'start predict idx={idx + 1}/{count} {filename}', end=True)
            # 处理输入图片
            ori_img = tools.convert_img(input_, args.channel_type)
            if is_use_420p:
                input_ = ori_img[:eval_size, :]
                copy_uv = ori_img[eval_size:, :]
            elif is_single_y:
                # 保留uv分量使用 label
                copy_uv = ori_img[:, :, 1:]
                input_ = ori_img[:, :, 0]
            else:
                input_ = ori_img
            real_input = input_

            if len(input_.shape) == 2:
                # 二维扩展4维
                input_ = input_[np.newaxis, :, :, np.newaxis]
            else:
                input_ = input_[np.newaxis, :]
            # 归一化
            input_ = input_ / 255.0
            input_ = tf.constant(input_, dtype=tf.float32)

            input_images = input_
            pre_outputs = model(input_images, args)

            # 需要对比的图片
            label_compare = tools.convert_img(label_ori, args.channel_type)
            if is_use_420p:
                label_ = label_compare[:eval_size, :]
            elif is_single_y:
                label_ = label_compare[:, :, 0]
            else:
                label_ = label_compare
            if len(label_.shape) == 2:
                # 若是转成了2维，扩展成3维
                label_ = label_[:, :, np.newaxis]

            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_model_path)

            time_ = time.time()
            result = sess.run(pre_outputs)
            cost = time.time() - time_

            base_name = f"{idx}-{filename.split('.')[0]}"
            out_img = tools.recover_to_img(result)

            psnr, ssim = compare.compare_images(label_, out_img, cost)
            high_quality = psnr > 39 or ssim > 0.92  # 仅存储高质量的
            log = "eval image: %d/%d, time: %.4f psnr: %.4f ssim: %.4f" % (idx, count, cost, psnr, ssim)
            utils.print_log(log, end=True)
            if save_dir:
                h_flag = 'high-' if high_quality else ''
                tools.save_img(real_input, f'{save_dir}/{base_name}--t1-in.jpg', args)
                tools.save_img(out_img, f'{save_dir}/{base_name}-t2-{h_flag}out.jpg', args)
                tools.save_img(label_, f'{save_dir}/{base_name}-t3-label.jpg', args)

            if args.eval_rgb and (is_use_420p or is_use_420p):
                if is_use_420p:
                    out_img_yuv = tools.y_to_yuv420p(args, out_img, copy_uv)
                    label_yuv = tools.y_to_yuv420p(args, label_, copy_uv)
                elif is_single_y:
                    out_img_yuv = tools.y_to_yuv(args, out_img, copy_uv)
                    label_yuv = tools.y_to_yuv(args, label_, copy_uv)
                # 和原图高清图对比
                psnr, ssim = compare2.compare_images(out_img_yuv, label_compare, cost)
                log = "compare by origin gt: %d/%d, time: %.4f psnr: %.4f ssim: %.4f" % (idx, count, cost, psnr, ssim)
                utils.print_log(log, end=True)
                if save_dir:
                    # 保存
                    tools.save_img(ori_img, f'{save_dir}/{base_name}-t4-in.jpg', args)
                    tools.save_img(out_img_yuv, f'{save_dir}/{base_name}-t5-y2bgr-out.jpg', args)
                    tools.save_img(label_yuv, f'{save_dir}/{base_name}-t6-y2bgr-label.jpg', args)

    utils.print_log('', end=True)
    utils.print_log(save_dir, checkpoint_model_path, end=True)
    compare.print()
    utils.print_log('y to rgb compare==>', end=True)
    compare2.print()
    if not args.local and save_dir:
        tools.zip_dir(save_dir)
        shutil.rmtree(save_dir)
