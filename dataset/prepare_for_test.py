#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import glob


class TestDataSet(object):
    """用于 test 时需要的"""

    def __init__(self, config):
        self.config = config
        if os.path.isfile(config.data_dir):
            file_paths = [config.data_dir]
        else:
            file_paths = glob.glob(os.path.join(config.data_dir, '*.png'))
            file_paths += glob.glob(os.path.join(config.data_dir, '*.PNG'))
            file_paths += glob.glob(os.path.join(config.data_dir, '*.jpg'))
            file_paths += glob.glob(os.path.join(config.data_dir, '*.jpeg'))
        self.img_items = sorted(file_paths)
        self.count = len(self.img_items)

    def __iter__(self):
        # 每次遍历重头开始
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.count:
            raise StopIteration
        path = self.img_items[self._index]
        image = cv2.imread(path)
        # 若不是验证，直接返回
        label = None
        input_image = image
        item = self._index, input_image, label, os.path.basename(path)
        self._index += 1
        return item
