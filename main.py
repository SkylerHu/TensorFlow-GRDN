#!/usr/bin/env python
# coding=utf-8
import os
import argparse

import utils
from utils import RunType, print_log
from dataset.tools import ChannelType

# 设置日志输出等级
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


parser = argparse.ArgumentParser(description='GRDN算法')
parser.add_argument('--run', type=str, default=RunType.TRAIN.value, help='运行')
parser.add_argument('--img_size', type=int, default=96, help='输入图片的大小')

parser.add_argument('--channel', type=int, default=3, help='图像通道channel数量')
parser.add_argument('--G0', type=int, default=64, help='开始卷积 feature maps数，RDN论文中给出的')
# G/D/C默认值是论文中只给出的实验值
# larger G contributes to better performance
parser.add_argument('--G', type=int, default=64, help='单个rdb feature map增长数量')
# larger D or C would lead to higher performance
parser.add_argument('--C', type=int, default=8, help=' 单个rdb的卷积层数')
parser.add_argument('--D', type=int, default=16, help='RDBs block的个数')
"""GRDN论文中，该默认值最佳"""
parser.add_argument('--grdbs_num', type=int, default=4, help='grdb的个数')
parser.add_argument('--rdbs_num', type=int, default=4, help='grdb中RDBs的个数')

parser.add_argument('--kernel_size', type=int, default=3, help='指定二维卷积窗口的高度和宽度')
parser.add_argument('--scale', type=int, default=1, help='对输入图片scale处理, GRDN无需处理')
parser.add_argument('--reduction', type=int, default=16, help='cbam压缩率')

parser.add_argument('--epochs', type=int, default=32, help='训练次数')
# 在一定范围内，一般来说 batch_size 越大，其确定的下降方向越准，引起训练震荡越小；增大到一定程度，其确定的下降方向已经基本不再变化。
parser.add_argument('--batch_size', type=int, default=16, help='每次梯度更新的样本数量')
# 选择合适的学习率，既不会收敛振荡也不会收敛太慢
parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率')
"""
tf.train.exponential_decay(
    learning_rate,  # 初始学习率
    global_step,    # 迭代次数
    decay_steps,    # 衰减周期，当staircase=True时，学习率在𝑑𝑒𝑐𝑎𝑦_𝑠𝑡𝑒𝑝𝑠内保持不变，即得到离散型学习率；
    decay_rate,     # 衰减率系数；
    staircase=False,# 是否定义为离散型学习率，默认False；
    name=None       # 名称，默认ExponentialDecay；
)
每训练 decay_steps 轮学习率乘以 decay_rate
"""
parser.add_argument('--decay_steps', type=float, default=4, help='tf衰减周期,乘以batch_n')
parser.add_argument('--decay_rate', type=float, default=0.5, help='tf衰减率系数')

parser.add_argument('--device', type=str, default=None, help='指定GPU')
parser.add_argument('--data_type', type=str, default=utils.DataType.SIDD.value, help='数据集来源')
parser.add_argument('--noise_sigma', type=int, default=50, help='噪声比[5,50]')
parser.add_argument('--keep_sigma', type=bool, default=False, help='是否固定噪声比训练')

parser.add_argument('--model', type=str, default='GRDN', help='算法类型')
parser.add_argument('--data_dir', type=str, default=None, help='数据集路径')
parser.add_argument('--resize', type=int, default=0, help='处理原始高清图片,图像目标高度')
parser.add_argument('--channel_type', type=str, default=ChannelType.RGB, help='训练的格式')
parser.add_argument('--dataset_count', type=int, default=0, help='自己指定数量')

parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='模型/检查点路径')
parser.add_argument('--record_dir', type=str, default='build', help='处理后的数据集路径')
parser.add_argument('--result_dir', type=str, default='result', help='保存结果的路径')
parser.add_argument('--step_save', type=int, default=10000, help='多少布的时候保存model一次')
parser.add_argument('--log_step', type=int, default=10, help='多少布输出日志')
parser.add_argument('--save_eval_dir', type=str, default=None, help='保存eval的图片路径')
parser.add_argument('--eval_size', type=int, default=256, help='验证的大小')
parser.add_argument('--eval_rgb', action='store_true', help='yuv按照rgb验证')
parser.add_argument('--local', action='store_true', help='是否本地,用于调试用')

args = parser.parse_args()

utils.init_logger(args)

if args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device


if args.run != RunType.TRAIN.value:
    args.batch_size = 1

if args.channel_type == ChannelType.YUV420P:
    args.channel = 1


print_log(args)

if __name__ == '__main__':
    if args.run == RunType.TRAIN.value:
        from train import do_train
        do_train(args)
    elif args.run == RunType.PREPARE.value:
        from dataset.prepare_sidd import prepare_ssid_data
        prepare_ssid_data(args)
    elif args.run == RunType.SAVE_PB.value:
        from eval import save_graph_to_pb
        save_graph_to_pb(args)
    elif args.run == RunType.EVAL_FILE.value:
        if not args.save_eval_dir:
            args.save_eval_dir = args.result_dir
        from eval_by_file import do_eval
        do_eval(args)
    elif args.run == RunType.EVAL.value:
        from eval import do_eval
        do_eval(args)
    elif args.run in RunType.TEST.value:
        from test import do_test
        do_test(args)
    elif args.run in RunType.RECORD.value:
        if not args.save_eval_dir:
            args.save_eval_dir = args.result_dir
        from record import do_test_record
        do_test_record(args)
    else:
        raise Exception(f'run={args.run}不合法')
