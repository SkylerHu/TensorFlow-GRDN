#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

# Must have `strides[0] = strides[3] = 1`
CONV_STRIDES = [1, 1, 1, 1]
CONV_ARGS = (CONV_STRIDES, 'SAME')


def gen_filter(filters, name=None):
    """
    :param filters: [filter_height, filter_width, in_channels, out_channels]
    :param name:
    :return:
    """
    # TODO 加上了 random_normal 会比较慢，暂时还没弄清楚用途
    # 不使用 Variable 还会报错
    return tf.Variable(tf.random_normal(filters, stddev=0.01), name=name)


def build_RDB(input_layer, config, flag):
    # 2.3 Local Residual Learning 单个RDB过程
    input_ = input_layer
    for i in range(config.C):
        # DenseLayer
        # 之前这里的输出错了，导致训练很慢，修复后正常; v1是备份的，有问题
        params_rdb = gen_filter(
            [config.kernel_size, config.kernel_size, config.G0 + i * config.G, config.G], name=f'params_rdb_{flag}_{i}')
        output = tf.nn.conv2d(input_, params_rdb, *CONV_ARGS, name=f'rdb_{flag}_{i}')
        output = tf.nn.relu(output)
        # 按照第3维进行拼接
        input_ = tf.concat([input_, output], axis=config.axis)
    # 2.2 LFF: Local Feature Fusion 指D个RDB输出结果Concat之后经过 1*1 Conv 与Conv layer之后得出 [公式] 的过程。
    # input filters: config.G0 + config.C * config.G
    params_lff = gen_filter([1, 1, config.G0 + config.C * config.G, config.G0], name=f'params_lff_{flag}')
    lff = tf.nn.conv2d(input_, params_lff, *CONV_ARGS, name=f'lff_{flag}')

    return tf.add(input_layer, lff)


def build_RDBs(input_layer, config, flag='rdbs'):
    rdb_concat = list()
    rdb_in = input_layer
    for idx in range(config.D):
        out = build_RDB(rdb_in, config, f'{flag}_{idx}')
        rdb_concat.append(out)
        rdb_in = out
    # 3. DFF: dense feature fusion 通过RDBs 抽取除了足够多了 Local Dense Features之后，进行全局信息整合
    fd = tf.concat(rdb_concat, axis=config.axis)
    params_gff = gen_filter([1, 1, config.G * config.D, config.G0], name=f'params_gff_{flag}')
    gff = tf.nn.conv2d(fd, params_gff, *CONV_ARGS, name=f'gff_{flag}')
    return gff


def build_UPNet(input_layer, config):
    params_up = gen_filter([config.kernel_size, config.kernel_size, config.G0, config.G0], name='params_up')
    out = tf.nn.conv2d(input_layer, params_up, *CONV_ARGS, name='con_up')
    out = tf.nn.relu(out)
    # 就算scale=1也要up sampling一下
    out = tf.keras.layers.UpSampling2D(size=(config.scale, config.scale))(out)
    return out


def build_RDN(input_layer, config):
    with tf.variable_scope('rdn'):
        # 1. SFENet: shallow feature extraction net 提取浅层特征
        # filters 整数，输出空间的维数(即在卷积中输出滤波器的数量), 此处认为是与input_shape的通道数一样，比如是RGB就是3
        # kernel_size 一个整数或2个整数的元组/列表，指定二维卷积窗口的高度和宽度
        # padding 卷积会导致输出图像越来越小，图像边界信息丢失，若想保持卷积后的图像大小不变，需要设置padding参数为same
        params_f_1 = gen_filter([config.kernel_size, config.kernel_size, config.channel, config.G0], name='params_f_1')
        f_1 = tf.nn.conv2d(input_layer, params_f_1, *CONV_ARGS, name='f_1')

        params_f0 = gen_filter([config.kernel_size, config.kernel_size, config.G0, config.G0], name='params_f0')
        f0 = tf.nn.conv2d(f_1, params_f0, *CONV_ARGS, name='f0')

        # 2. RDBs: redidual dense blocks 多个密集残差块(Residual Dense Blocks (RDBs))个RDB。
        # 2.1 Contiguous memory: 意思就是dense connection，连续的连接保证了连续的低级高级信息存储和记忆，
        # 每一个RDB模块的输出结果，concate上一个RDB模块的输出，以及所有的卷积层之间的信息，
        # 其包含local feature极其丰富，也包含了层级信息，保证了信息不丢失。
        gff = build_RDBs(f0, config)

        # 3.1 GFF: Global feature fusion 全局特征融合: rdbs+fgf
        params_fgf = gen_filter([config.kernel_size, config.kernel_size, config.G0, config.G0], name='params_fgf')
        fgf = tf.nn.conv2d(gff, params_fgf, *CONV_ARGS, name='fgf')

        fdf = tf.add(fgf, f_1)
        # 4. UPNet: up-sampling net
        upn = build_UPNet(fdf, config)

        params_ihr = gen_filter([config.kernel_size, config.kernel_size, config.G0, config.channel], name='params_ihr')
        ihr = tf.nn.conv2d(upn, params_ihr, *CONV_ARGS, name='ihr')
        return ihr
