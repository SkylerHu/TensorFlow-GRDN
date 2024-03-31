#!/usr/bin/env python
# coding=utf-8
import os
import json
import datetime

import tensorflow as tf

import utils
from utils import print_log, DataType


def get_record_path(args, count=None, is_eval=False):
    """
    :param args:
    :param count: ==0时候，是prepare时需要的
    :param is_eval: 是否是用于验证
    :return:
    """
    dir_name = f'{args.record_dir}/records'
    if args.data_type != DataType.SIDD.value:
        dir_name = f'{args.record_dir}/records-{args.data_type}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = f'c{args.channel}-s{args.img_size}x{args.img_size}'
    if args.resize > 0:
        filename = f'rs{args.resize}-{filename}'
    if args.channel_type:
        filename = f'{args.channel_type}-{filename}'
    filename = f'dataset-{filename}'

    if is_eval:
        filename = f'{filename}-eval'
    if count is not None:
        filename = f'count-{count}-{filename}'
    return f'{dir_name}/{filename}.tfrecords'


def set_record_info(args, shape):
    """记录图片转换后的大小"""
    info_dir = os.path.dirname(get_record_path(args))
    file_name = os.path.join(info_dir, f'{args.img_size}x{args.img_size}-{args.channel_type}.txt')
    if os.path.exists(file_name):
        os.remove(file_name)
    w, h = shape
    with open(file_name, 'w') as f:
        f.write(f'{w} {h}')


def get_record_info(args):
    info_dir = os.path.dirname(get_record_path(args))
    file_name = os.path.join(info_dir, f'{args.img_size}x{args.img_size}-{args.channel_type}.txt')
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            line = f.read()
        w, h = line.strip().split(' ')
        w, h = int(w), int(h)
    else:
        w, h = args.img_size, args.img_size
    return w, h


def gen_checkpoint_dir(args):
    model_name = f'{args.model}_{args.channel_type}_c{args.channel}_{args.grdbs_num}_{args.rdbs_num}_D{args.D}_' \
                 f'{args.G0}_G{args.G}_C{args.C}_x{args.scale}'
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def save_model(args, saver, sess, global_step=None, **kwargs):
    checkpoint_dir = gen_checkpoint_dir(args)

    print_log(f'start dir is {checkpoint_dir} ...')
    file_path = os.path.join(checkpoint_dir, 'train.model')
    saver.save(sess, file_path, **kwargs)
    print_log('save ckpt done !!!')
    # 记录步数
    step_file = os.path.join(checkpoint_dir, 'step.txt')
    with open(step_file, 'w') as f:
        f.write(json.dumps(global_step))


def get_checkpoint_path_and_step(args):
    checkpoint_dir = gen_checkpoint_dir(args)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    _path, _step = None, 0
    if ckpt and ckpt.model_checkpoint_path:
        _path = str(ckpt.model_checkpoint_path)
        try:
            step_file = os.path.join(os.path.dirname(_path), 'step.txt')
            with open(step_file, 'r') as f:
                content = f.read()
            _step = json.loads(content)
        except Exception:
            pass
    return _path, _step


def gen_summary_dir(args):
    checkpoint_dir = gen_checkpoint_dir(args)
    path = os.path.join(checkpoint_dir, 'log')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_model_func(args):
    from .grdn import build_GRDN
    model = build_GRDN
    return model


def get_result_path(args):
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M-%S')
    tag = f'{args.channel_type}-{now}' if args.channel_type else now
    save_dir = os.path.join(args.save_eval_dir, args.model, tag)
    return save_dir


def image_to_noise(args, labels_images):
    """给图片增加噪声"""
    shape = labels_images.get_shape().as_list()
    _, w, h, c = shape

    if args.run == utils.RunType.TRAIN.value and not args.keep_sigma:
        # 随机噪声比
        stdn = tf.random.uniform([args.batch_size, 1, 1, 1], minval=5 / 255.0, maxval=50 / 255.0, dtype=tf.float32)
    else:
        stdn = args.noise_sigma / 255.0

    # 初始化噪声图片
    noise = tf.broadcast_to(stdn, [args.batch_size, w, h, c])
    noise = tf.random_normal(shape=tf.shape(labels_images), mean=0.0, stddev=noise, dtype=tf.float32)
    input_images = labels_images + noise
    # 噪声map
    noise_map = tf.broadcast_to(stdn, (args.batch_size, w, h, 1))
    return input_images, noise_map
