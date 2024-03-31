#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

import utils
from model import helper
from dataset import tools
from dataset.parse import gen_paras_tf_func


def do_test_record(args):
    record_file_path = helper.get_record_path(args, count=args.dataset_count)
    dataset = tf.data.TFRecordDataset(record_file_path)

    _parse_tf_example = gen_paras_tf_func(args)

    batch_size = 1
    dataset = dataset.map(_parse_tf_example)
    # 将数据集中连续的元素以batch_size为单位集合成批次
    dataset = dataset.batch(batch_size)
    # prefetch 预取数据，即它总是使得一定批次的数据准备被加载。
    dataset = dataset.prefetch(batch_size)

    save_dir = helper.get_result_path(args)

    utils.print_log(f'load {record_file_path} ', end=True)
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        if args.data_type == utils.DataType.DAVIS.value:
            labels_images = iterator.get_next()
            input_images, noise_map = helper.image_to_noise(args, labels_images)
        else:
            input_images, labels_images = iterator.get_next()

        step = 0
        while True:
            step += 1
            print(f'step is {step}')
            try:
                input_img, label_img = sess.run([input_images, labels_images])
                tools.save_output_img(input_img, f'{save_dir}/{step}-input.jpg', args)
                tools.save_output_img(label_img, f'{save_dir}/{step}-label.jpg', args)
            except tf.errors.OutOfRangeError:
                break
