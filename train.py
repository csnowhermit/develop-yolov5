import os
import cv2
import glob
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import config


if __name__ == '__main__':
    weights_path = './checkpoint' + os.sep  # weights dir
    last = weights_path + 'last.pt'
    best = weights_path + 'best.pt'
    results_file = 'results.txt'

    mixed_precision = True    # 混合精度，需要用到NVIDIA apex包
    if config.device.type == 'cpu':
        mixed_precision = False    # cpu不允许混合精度

    # 初始化随机数种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 读取数据集配置文件
    with open(config.data, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = int(data_dict['nc'])

    # 删除以前的训练结果
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)



