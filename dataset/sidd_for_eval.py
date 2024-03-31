#!/usr/bin/env python
# coding=utf-8
import os
import cv2

from utils import print_log


def ssid_eval_data(data_dir, config=None):
    # 裁剪按照 短边进行计算
    corp_x, corp_y, stride = config.eval_size, config.eval_size, config.eval_size
    # 找出所有噪点的图片
    command = f"find {data_dir} -name '*NOISY_*.PNG' -type f"
    file_paths = os.popen(command).readlines()

    count = 0
    for f_idx, img_path in enumerate(sorted(file_paths)):
        print_log(f'{f_idx}/{len(file_paths)} {img_path} init ...', end=True)
        img_path = img_path.strip()
        file_name = os.path.basename(img_path)
        name, ext = file_name.split('.')
        # 都按照bgr直接读取，裁剪后再转换格式
        # 读取有噪点的图片, 输入
        img_noisy = cv2.imread(img_path)
        # 输出
        img_gt = cv2.imread(img_path.replace('NOISY_', 'GT_'))
        if img_gt.shape[:2] != img_noisy.shape[:2]:
            continue
        h, w = img_gt.shape[:2]  # shape有可能是2维，也可能是3维

        x_locs = list(range(0, h - corp_y + 1, stride))
        y_locs = list(range(0, w - corp_x + 1, stride))
        for x_idx, x in enumerate(x_locs):  # +1 是为了处理正好h=n*corp_x的时候
            for y_idx, y in enumerate(y_locs):
                if x + corp_x > h or y + corp_y > w:
                    continue
                input_image = img_noisy[x: x + corp_x, y: y + corp_y]
                label_image = img_gt[x: x + corp_x, y: y + corp_y]
                print_log(f'{f_idx}/{len(file_paths)} {count}/{x}*{y} yield ...', end=True)
                yield count, input_image, label_image, f"{name}_{x}_{y}.{ext}"
                count += 1
