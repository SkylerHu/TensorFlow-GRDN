#!/usr/bin/env python
# coding=utf-8
import time

import tensorflow as tf

import utils
from utils import print_log
from model import helper
from dataset.parse import gen_paras_tf_func


def do_train(args):
    if not args.dataset_count:
        raise Exception('参数 dataset_count 必不可少')
    dataset_count = args.dataset_count

    model = helper.get_model_func(args)
    record_file_path = helper.get_record_path(args, count=args.dataset_count)
    dataset = tf.data.TFRecordDataset(record_file_path)

    _parse_tf_example = gen_paras_tf_func(args)

    # 打乱顺序
    dataset = dataset.shuffle(dataset_count)
    dataset = dataset.map(_parse_tf_example)
    # 将数据集中连续的元素以batch_size为单位集合成批次
    dataset = dataset.batch(args.batch_size)
    # prefetch 预取数据，即它总是使得一定批次的数据准备被加载。
    dataset = dataset.prefetch(args.batch_size)

    # 找之前训练的结果
    checkpoint_path, step = helper.get_checkpoint_path_and_step(args)
    print_log(f'load {record_file_path} checkpoint {checkpoint_path} step={step}', end=True)
    with tf.Session() as sess:
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        batch_num = dataset_count // args.batch_size + (1 if dataset_count % args.batch_size != 0 else 0)

        if args.data_type == utils.DataType.DAVIS.value:
            labels_images = iterator.get_next()
            input_images, noise_map = helper.image_to_noise(args, labels_images)
            predicted = model(input_images, args, noise_map=noise_map)
        else:
            input_images, labels_images = iterator.get_next()
            predicted = model(input_images, args)

        # 定义损失函数； 也有人用其他两种
        loss = tf.reduce_mean(tf.square(labels_images - predicted))  # 均方差函数
        # loss = tf.reduce_mean(tf.abs(labels_images - predicted))
        # loss = tf.reduce_mean(tf.losses.absolute_difference(labels_images, predicted))
        # loss = 1 - tf.reduce_mean(tf.image.ssim(labels_images, predicted, max_val=1.0))

        # 全局步骤
        global_step = tf.Variable(step, trainable=False)
        # 指数衰减学习率
        learning_rate = tf.train.exponential_decay(
            args.learning_rate, global_step,
            # 没有指定 decay_steps 时，按照一个迭代衰减10次
            int(args.decay_steps * batch_num), args.decay_rate,
            staircase=True
        )
        print_log('init optimizer', end=True)
        # 寻找全局最优点的优化算法，实现自动梯度下降
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        learning_step = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        # tf.summary.scalar('ssim_loss', ssim_loss)

        # 定义写入磁盘操作：可以将所有summary全部保存到磁盘，以便tensorboard显示
        merged_summary_op = tf.summary.merge_all()

        print_log('init Saver', end=True)
        saver = tf.train.Saver()

        # 恢复之前训练的
        if checkpoint_path:
            saver.restore(sess, checkpoint_path)
        else:
            # 不是恢复的，才需要初始化参数
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(helper.gen_summary_dir(args), graph=sess.graph)

        cost = 0
        mid_cost = 0
        for epoch in range(args.epochs):
            print_log('init iterator.initializer', end=True)
            sess.run(iterator.initializer)
            idx = 0
            print_log(f'start epoch={epoch + 1}', end=True)
            while True:
                step += 1
                idx += 1
                flag = f'Epoch/step {epoch + 1}/{idx}'
                start = time.time()
                try:
                    # mid = time.time()
                    _loss, lr, _, summary_str = sess.run([loss, learning_rate, learning_step, merged_summary_op])
                    # _loss, lr, summary_str = sess.run([loss, learning_rate, merged_summary_op])
                    summary_writer.add_summary(summary_str, global_step=step)
                except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
                    # InvalidArgumentError 主要是最后一个不匹配
                    helper.save_model(args, saver, sess, global_step=step)
                    print_log(f'{flag} OutOfRangeError ~', end=True)
                    break
                cost += time.time() - start
                # mid_cost += time.time() - mid
                if args.device is None or step % args.log_step == 0:
                    print_log(
                        "Epoch: [%4d], batch: [%6d/%6d], loss: [%.18f],"
                        "\t lr: [%.9f], step: [%d] mid: %.3f cost: %.3f" % (
                            (epoch + 1), idx, batch_num, _loss, lr, step, cost, mid_cost,
                        ), end=True,
                    )
                    # 将事件文件刷新到磁盘
                    summary_writer.flush()

                    cost = 0
                    mid_cost = 0
                if step % args.step_save == 0:
                    helper.save_model(args, saver, sess, global_step=step)
        # 最后保存一次
        helper.save_model(args, saver, sess, global_step=step)

        # 将事件文件刷新到磁盘并关闭该文件
        summary_writer.close()
