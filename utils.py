#!/usr/bin/env python
# coding=utf-8
import os
import logging
import datetime

from enum import Enum


logger = None


def init_logger(args):
    global logger
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M-%S')
    model_name = f'{args.model}_{args.channel_type}_c{args.channel}_{args.grdbs_num}_{args.rdbs_num}_D{args.D}_' \
                 f'{args.G0}_G{args.G}_C{args.C}_x{args.scale}'
    file_name = f'build/logs/train-{model_name}.log.{now}'
    log_dir = os.path.dirname(file_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('app')
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
        filename=file_name, filemode='w',
    )


def print_log(*args, end=False):
    global logger
    logger.info(' '.join([str(i) for i in args]))
    if not end:
        print(*args, end='\r', flush=True)
    else:
        print(*args, flush=True)


class RunType(Enum):
    """"运行类型"""
    PREPARE = 'prepare'  # 对输入的图片进行参数初始化
    TRAIN = 'train'  # 训练
    EVAL = 'eval'  # 验证model，使用prepare的TFRecord作为输入
    EVAL_FILE = 'eval_file'  # 验证model,输入直接读取图像文件
    TEST = 'test'  # 测试使用
    SAVE_PB = 'save_pb'  # 保存为pb文件，可以供ffmpeg使用
    RECORD = 'record'


class DataType(Enum):
    SIDD = 'sidd'
    DAVIS = 'davis'  # 其他数据集
