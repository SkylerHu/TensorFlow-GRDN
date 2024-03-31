#!/usr/bin/env python
# coding=utf-8
import time
import shutil

import tensorflow as tf

import utils
from model import helper
from dataset import tools
from dataset.parse import gen_paras_tf_func


def do_eval(args):
    checkpoint_model_path, _ = helper.get_checkpoint_path_and_step(args)

    model = helper.get_model_func(args)

    count = args.dataset_count

    _parse_tf_example = gen_paras_tf_func(args)
    data_path = helper.get_record_path(args, count=count, is_eval=True)
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_tf_example)
    dataset = dataset.batch(args.batch_size)

    # 存储位置相关
    save_dir = helper.get_result_path(args) if args.save_eval_dir else None
    utils.print_log('eval save_dir is', save_dir, end=True)

    idx = 0
    compare = tools.CompareInfo()
    with tf.Session() as sess:
        iterator = dataset.make_one_shot_iterator()
        input_ts, label_ts = iterator.get_next()
        pre_outputs = model(input_ts, args)

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_model_path)

        while True:
            try:
                utils.print_log(f'eval start predict idx={idx}')

                time_ = time.time()
                input_, label_, result = sess.run([input_ts, label_ts, pre_outputs])
                cost = time.time() - time_

                # 4维去掉第一维
                label_img = tools.recover_to_img(label_)
                out_img = tools.recover_to_img(result)

                # compare_label, compare_out = label_img, out_img
                # 就按照训练格式进行对比，特别是yuv和rgb互相转换有损
                # 转变成rgb进行对比
                # if args.channel_type == ChannelType.BGR.value:
                #     compare_label = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
                #     compare_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                # elif args.channel_type == ChannelType.YUV.value:
                #     compare_label = cv2.cvtColor(label_img, cv2.COLOR_YUV2RGB)
                #     compare_out = cv2.cvtColor(out_img, cv2.COLOR_YUV2RGB)
                # elif args.channel_type == ChannelType.YUV420P.value:
                #     # 按照rgb进行对比
                #     compare_label = cv2.cvtColor(label_img, cv2.COLOR_YUV2RGB_I420)
                #     compare_out = cv2.cvtColor(out_img, cv2.COLOR_YUV2RGB_I420)

                # import pdb; pdb.set_trace()
                psnr, ssim = compare.compare_images(label_img, out_img, cost)

                log = "eval image: %d/%d, time: %.4f psnr: %.4f ssim: %.4f" % (idx, count, cost, psnr, ssim)
                utils.print_log(log, end=True)
            except tf.errors.OutOfRangeError:
                utils.print_log('eval OutOfRangeError ~', end=True)
                break

            idx += 1

    utils.print_log('', end=True)
    utils.print_log('eval checkpoint_model_path is', checkpoint_model_path, end=True)
    compare.print()
    if save_dir:
        tools.zip_dir(save_dir)
        shutil.rmtree(save_dir)


def save_graph_to_pb(args):
    model = helper.get_model_func(args)
    # 输入的高宽不确定用None
    input_shape = [1, None, None, args.channel]
    # 定义变量
    with tf.Session() as sess:
        input_images = tf.placeholder(tf.float32, input_shape, name='x')
        predicted = model(input_images, args)
        # 命令
        tf.identity(predicted, name='y')
        model_checkpoint_path, _ = helper.get_checkpoint_path_and_step(args)
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)
        # 将模型文件保存为 *.pb，这个文件包括了一切信息，将被ffmpeg所使用
        graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['y'])
        checkpoint_dir = helper.gen_checkpoint_dir(args)
        name = f'{args.model.upper()}.pb'
        tf.train.write_graph(graph_def, checkpoint_dir, name, as_text=False)
    utils.print_log('pb file is', f'{checkpoint_dir}/{name}', end=True)
