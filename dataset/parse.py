#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

import utils
from model.helper import get_record_info


def gen_paras_tf_func(args):
    # w, h = get_record_info(args)
    w, h = args.img_size, args.img_size
    # 定义输入输出 格式
    input_shape = [w, h, args.channel]
    output_shape = [w * args.scale, h * args.scale, args.channel]

    def _parse_davis_example(example_proto):
        # 定义一个特征词典，和写TFRecords时的特征词典相对应
        features = {
            'label': tf.FixedLenFeature([], tf.string),
        }
        # 根据上面的特征解析单个数据
        parsed_features = tf.parse_single_example(example_proto, features)
        _label = tf.decode_raw(parsed_features['label'], tf.uint8)
        # 转换
        _label = tf.cast(tf.reshape(_label, output_shape), tf.float32)

        # 归一化
        _label = _label / 255.0
        return _label

    def _parse_tf_example(example_proto):
        # 定义一个特征词典，和写TFRecords时的特征词典相对应
        features = {
            'input': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        }
        # 根据上面的特征解析单个数据
        parsed_features = tf.parse_single_example(example_proto, features)
        _input, _label = parsed_features['input'], parsed_features['label']

        _input, _label = tf.decode_raw(_input, tf.uint8), tf.decode_raw(_label, tf.uint8)
        # 转换
        _input, _label = tf.reshape(_input, input_shape), tf.reshape(_label, output_shape)
        # 归一化
        _input, _label = tf.cast(_input, tf.float32) / 255.0, tf.cast(_label, tf.float32) / 255.0
        return _input, _label

    if args.data_type == utils.DataType.DAVIS.value:
        return _parse_davis_example

    return _parse_tf_example
