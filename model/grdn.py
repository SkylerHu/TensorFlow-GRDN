#!/usr/bin/env python
# coding=utf-8
from .rdn import *
from .cbam import cbam_net

# 下采样
DOWN_UP_SCALE = 2  # 放大缩小的倍数
CONV_DOWN_STRIDES = [1, DOWN_UP_SCALE, DOWN_UP_SCALE, 1]
CONV_DOWN_ARGS = (CONV_DOWN_STRIDES, 'SAME')


def build_GRDB(input_layer, config, flag):
    rdb_concat = list()
    in_ = input_layer
    for i in range(config.rdbs_num):
        out = build_RDB(in_, config, flag=f'{flag}_{i}')
        rdb_concat.append(out)
        in_ = out
    rdbs = tf.concat(rdb_concat, axis=config.axis)
    # 1x1 convolutional layer
    params_grdb = gen_filter([1, 1, config.G0 * config.rdbs_num, config.G0], name=f'params_grdb_{flag}')
    grdb = tf.nn.conv2d(rdbs, params_grdb, *CONV_ARGS, name=f'grdb_{flag}')
    out = tf.add(input_layer, grdb)
    return out


def build_GRDN(input_layer, config, **kwargs):
    setattr(config, 'axis', -1)  # 要求输入格式要求是rgb

    with tf.variable_scope('grdn'):
        # 关于kernel_size，除了up/down是4x4 filter, rdb contact用1x1，其余都是3x3

        # 1. SFENet: shallow feature extraction net 提取浅层特征
        # filters 整数，输出空间的维数(即在卷积中输出滤波器的数量), 此处认为是与input_shape的通道数一样，比如是RGB就是3
        # kernel_size 一个整数或2个整数的元组/列表，指定二维卷积窗口的高度和宽度
        # padding 卷积会导致输出图像越来越小，图像边界信息丢失，若想保持卷积后的图像大小不变，需要设置padding参数为same
        params_f_1 = gen_filter([config.kernel_size, config.kernel_size, config.channel, config.G0], name='params_f_1')
        f_1 = tf.nn.conv2d(input_layer, params_f_1, *CONV_ARGS, name='f_1')

        up_down_filter = 4
        params_f0 = gen_filter([up_down_filter, up_down_filter, config.G0, config.G0], name='params_f0')
        f0 = tf.nn.conv2d(f_1, params_f0, *CONV_DOWN_ARGS, name='f0')

        # 2. RDBs: redidual dense blocks 多个密集残差块(Residual Dense Blocks (RDBs))个RDB。
        # 2.1 Contiguous memory: 意思就是dense connection，连续的连接保证了连续的低级高级信息存储和记忆，
        # 每一个RDB模块的输出结果，concate上一个RDB模块的输出，以及所有的卷积层之间的信息，
        # 其包含local feature极其丰富，也包含了层级信息，保证了信息不丢失。
        # 处理多层grdbs
        out_grdbs = f0
        for idx in range(config.grdbs_num):
            out_grdbs = build_GRDB(out_grdbs, config, idx)

        # 4. UPNet: up-sampling net 反卷积
        upn = tf.keras.layers.Conv2DTranspose(
            config.G0, up_down_filter, (DOWN_UP_SCALE, DOWN_UP_SCALE), 'SAME')(out_grdbs)
        # 除了直接使用 convXdTranspose，还可以分两步实现反卷积。即：第一步上采样，第二步正常的卷积操作; 但是有区别
        # upn = tf.keras.layers.UpSampling2D(size=(DOWN_UP_SCALE, DOWN_UP_SCALE))(out_grdbs)
        # up_cov = tf.nn.conv2d(upn, params_f0, *CONV_ARGS, name='up_cov')

        cbam = cbam_net(upn, config)

        params_ihr = gen_filter([config.kernel_size, config.kernel_size, config.G0, config.channel], name='params_ihr')
        ihr = tf.nn.conv2d(cbam, params_ihr, *CONV_ARGS, name='ihr')

        out = tf.add(input_layer, ihr)
        return out
