#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

from .rdn import gen_filter, CONV_ARGS


def cbam_net_v1(input_layer, config):
    channel = config.G0
    reduce_channel = channel // config.reduction

    # channel attention
    x_mean = tf.reduce_mean(input_layer, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, reduce_channel, 1, activation=tf.nn.relu, name='mean_CA1')  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1, name='mean_CA2')   # (B, 1, 1, C)

    x_max = tf.reduce_max(input_layer, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, reduce_channel, 1, activation=tf.nn.relu, name='max_CA1')  # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1, name='max_CA2')  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)  # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input_layer, x)   # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=config.axis)     # (B, W, H, 2)
    y = tf.layers.conv2d(
        y, 1, 7, padding='same', activation=tf.nn.sigmoid, name='spat_a')  # (B, W, H, 1)
    y = tf.multiply(x, y)  # (B, W, H, C)

    return y


def cbam_net(input_layer, config):
    """
    @Convolutional Block Attention Module
    """
    # 此处channel不同于图像的通道数量，是filter里的input_channel
    channel = config.G0
    reduce_channel = channel // config.reduction
    kernel_size = 1

    # channel attention
    # mean
    x_mean = tf.reduce_mean(input_layer, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    params_ca1 = gen_filter([kernel_size, kernel_size, channel, reduce_channel], name='p_mean_CA1')
    x_mean = tf.nn.conv2d(x_mean, params_ca1, *CONV_ARGS, name='mean_CA1')  # (B, 1, 1, C // r)
    x_mean = tf.nn.relu(x_mean)

    params_ca2 = gen_filter([kernel_size, kernel_size, reduce_channel, channel], name='p_mean_CA2')
    x_mean = tf.nn.conv2d(x_mean, params_ca2, *CONV_ARGS, name='mean_CA2')  # (B, 1, 1, C)

    # max
    x_max = tf.reduce_max(input_layer, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    p_max_ca1 = gen_filter([kernel_size, kernel_size, channel, reduce_channel], name='p_max_CA1')
    x_max = tf.nn.conv2d(x_max, p_max_ca1, *CONV_ARGS, name='max_CA1')  # (B, 1, 1, C // r)
    x_max = tf.nn.relu(x_max)
    p_max_ca2 = gen_filter([kernel_size, kernel_size, reduce_channel, channel], name='p_max_CA2')
    x_max = tf.nn.conv2d(x_max, p_max_ca2, *CONV_ARGS, name='max_CA2')  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)  # (B, 1, 1, C)
    # 将输出压缩至0～1的范围
    x = tf.nn.sigmoid(x)  # (B, 1, 1, C)
    x = tf.multiply(input_layer, x)  # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=config.axis)  # (B, W, H, 2)

    kernel_size = 7  # 论文中表示这个最优
    p_sp_a = gen_filter([kernel_size, kernel_size, 2, 1], name='p_sp_a')
    y = tf.nn.conv2d(y, p_sp_a, *CONV_ARGS, name='sp_a')  # (B, W, H, 1)
    y = tf.nn.sigmoid(y)

    y = tf.multiply(x, y)  # (B, W, H, C)

    return y
