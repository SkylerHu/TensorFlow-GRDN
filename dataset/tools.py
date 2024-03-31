#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import enum
import zipfile
import numpy as np
import tensorflow as tf

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import utils


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ChannelType(enum.Enum):
    BGR = 'bgr'
    RGB = 'rgb'
    YUV = 'yuv'
    YUV420P = 'yuv420p'


def read_img(path, config):
    img = cv2.imread(path)
    # im read 默认为 BGR 格式
    return convert_img(img, config.channel_type)


def convert_img(img, channel_type):
    # 输入img必须是 bgr
    if channel_type == ChannelType.RGB.value:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif channel_type == ChannelType.YUV.value:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif channel_type == ChannelType.YUV420P.value:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    return img


def save_output_img(image, path, config):
    image = recover_to_img(image)
    save_img(image, path, config)


def save_img(img, path, config):
    if config.channel_type == ChannelType.RGB.value:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif config.channel_type == ChannelType.YUV.value:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    elif config.channel_type == ChannelType.YUV420P.value:
        h, w = img.shape[:2]
        if h != w:
            # 其实是 h = w + w / 2 不等的时候，只有y通道
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
        else:
            if img.shape == 3:
                img = img[:, :, 0]

    p_dir = os.path.dirname(path)
    if not os.path.isdir(p_dir):
        os.makedirs(p_dir)

    cv2.imwrite(path, img)


def recover_to_img(output):
    """四位输出 转回 图片"""
    img = np.squeeze(output, axis=0) * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def normalization_img(img, shape):
    """标准 & 归一化数据"""
    output = img.reshape(shape)
    output = output / 255.0  # 归一化
    return output


def _random_handle_img(img, values, w_is_h):
    """随机对图片进行水平或者旋转翻转"""
    flip_v, rot_v = values
    out = img
    # 翻转
    if flip_v < 0.3:
        out = np.flip(img, 0)
    elif flip_v > 0.7:
        out = np.flip(img, 1)
    # 旋转
    if rot_v < 0.5:
        if w_is_h:
            out = np.rot90(out, 1, (0, 1))
        else:
            # 不登高，需要旋转2次
            out = np.rot90(out, 2, (0, 1))
    return out


def _standard_handle_img(input_image, label_image, config):
    # 加入随机因素
    values = np.random.randint(10, size=2)
    w_is_h = config.img_size == config.img_size
    input_image = _random_handle_img(input_image, values, w_is_h)
    label_image = _random_handle_img(label_image, values, w_is_h)

    input_image = convert_img(input_image, config.channel_type)
    label_image = convert_img(label_image, config.channel_type)

    if config.channel == 1:
        if config.channel_type == ChannelType.YUV420P.value:
            # yuv420p 使用平面，只有一维
            input_image = input_image[:config.img_size, :]
            label_image = label_image[:config.img_size, :]
        elif len(input_image.shape) == 3:
            input_image = input_image[:, :, 0]
            label_image = label_image[:, :, 0]
    return input_image, label_image


def crop_image(image, size, scale, channel, x, y):
    corp_size = size * scale
    # 裁切正方形
    label_image = image[x: x + corp_size, y: y + corp_size]
    # 输入图按照高清图缩小
    input_image = cv2.resize(label_image, (size, size), interpolation=cv2.INTER_CUBIC)
    return _standard_handle_img(input_image, label_image, channel)


def mod_crop(img, scale=3):
    """ 原图进行裁边，标准化 图片输入 """
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w]
    return img


def crop_sidd_image(img_noisy, img_gt, corp_size, x, y, config):
    corp_x, corp_y = corp_size
    input_image = img_noisy[x: x + corp_x, y: y + corp_y]
    label_image = img_gt[x: x + corp_x, y: y + corp_y]
    return _standard_handle_img(input_image, label_image, config)


def crop_single_image(img_gt, corp_size, x, y, config):
    corp_x, corp_y = corp_size
    input_image = img_gt[x: x + corp_x, y: y + corp_y]
    # 加入随机因素
    values = np.random.randint(10, size=2)
    w_is_h = config.img_size == config.img_size
    input_image = _random_handle_img(input_image, values, w_is_h)
    input_image = convert_img(input_image, config.channel_type)
    return input_image


def zip_dir(path):
    """
    压缩指定文件夹
    """
    # 按照文件名字直接命名
    zip_filename = os.path.join(os.path.dirname(path), os.path.basename(path) + '.zip')
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path, _, filenames in os.walk(path):
            # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
            fpath = path.replace(path, '')
            for filename in filenames:
                zip_file.write(os.path.join(path, filename), os.path.join(fpath, filename))


def y_to_yuv(config, img_y, uv):
    """扩展Y 变成 uv"""
    if len(img_y.shape) == 3:
        # 变成2维
        img_y = np.squeeze(img_y, axis=2)
    # 把uv信息和Y信息合并，然后保存起来
    out_img = np.insert(uv, 0, img_y, axis=2)
    return out_img


def y_to_yuv420p(config, img_y, uv):
    if len(img_y.shape) == 3:
        img_y = img_y[:, :, 0]
    if len(uv.shape) == 3:
        uv = uv[:, :, 0]
    yuv420p = np.concatenate((img_y, uv), axis=0)
    return yuv420p


class CompareInfo(object):

    def __init__(self):
        self.count = 0
        self.sum_time = 0
        self.sum_time, self.sum_psnr, self.min_psnr, self.max_psnr = 0, 0, 100, 0
        self.sum_ssim, self.min_ssim, self.max_ssim = 0, 1, 0

    def compare_images(self, in_img, out_img, cost):
        self.count += 1
        self.sum_time += cost
        psnr = peak_signal_noise_ratio(in_img, out_img)
        if self.min_psnr > psnr:
            self.min_psnr = psnr
        if self.max_psnr < psnr:
            self.max_psnr = psnr
        self.sum_psnr += psnr
        is_multi = len(out_img.shape) > 2
        ssim = structural_similarity(in_img, out_img, multichannel=is_multi)
        if self.min_ssim > ssim:
            self.min_ssim = ssim
        if self.max_ssim < ssim:
            self.max_ssim = ssim
        self.sum_ssim += ssim

        return psnr, ssim

    def print(self):
        if self.count > 0:
            utils.print_log(f"eval count = {self.count} ,  Avg. Time:", self.sum_time / self.count, end=True)
            utils.print_log(f"eval PSNR Range: {self.min_psnr} , {self.max_psnr}", end=True)
            utils.print_log(f"eval SSIM Range: {self.min_ssim} , {self.max_ssim}", end=True)
            utils.print_log("eval Avg PSNR:", self.sum_psnr / self.count, end=True)
            utils.print_log("eval Avg SSIM:", self.sum_ssim / self.count, end=True)
        else:
            utils.print_log(f"eval count = {self.count}", end=True)
