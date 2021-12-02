import os
import cv2
import glob
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

import config
from models.yolo import Model
from utils.dataset import LoadImageAndLabels
from utils.util import labels_to_class_weights
from utils import torch_utils


if __name__ == '__main__':
    weights_path = './checkpoint' + os.sep  # weights dir
    last = weights_path + 'last.pt'
    best = weights_path + 'best.pt'
    results_file = 'results.txt'

    # 初始化随机数种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # 构建模型
    model = Model().to(config.device)
    accumulate = max(round(64 / config.batch_size), 1)  # 优化前累计损失
    config.hyp['weight_decay'] *= config.batch_size * accumulate / 64    # 权重衰减
    param_group0, param_group1, param_group2 = [], [], []    # 分组优化
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                param_group2.append(v)
            elif '.weight' in k and '.bn' not in k:
                param_group1.append(v)
            else:
                param_group0.append(v)

    # 带动量的SGD
    optimizer = optim.SGD(param_group0, lr=config.hyp['lr0'], momentum=config.hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': param_group1, 'weight_decay': config.hyp['weight_decay']})    # 权重衰减
    optimizer.add_param_group({'params': param_group2})
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(param_group2), len(param_group1), len(param_group0)))
    del param_group0, param_group1, param_group2

    # 加载预训练模型
    if config.weight.endswith(".pt") is True:
        checkpoint = torch.load(config.weight, map_location=config.device)

        try:
            # 预训练模型要跟定义的模型能对上
            checkpoint['model'] = {k: v for k, v in checkpoint['model'].state_dict().items() if
                                   model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(checkpoint['model'], strict=False)
        except KeyError as e:
            raise KeyError("预训练模型与定义的模型不匹配") from e

        # 优化器的参数
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_fitness = checkpoint['best_fitness']

        # 上一次的训练结果
        if checkpoint['training_results'] is not None:
            with open(results_file, 'w') as f:
                f.write(checkpoint['training_results'])
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint

        # 用scheduler调整优化器
        lf = lambda x: (((1 + math.cos(x * math.pi / config.epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1

        # 加载数据集
        train_dataset = LoadImageAndLabels(img_path=config.train_dataset,
                                     img_size=config.img_size,
                                     batch_size=config.batch_size,
                                     hyp=config.hyp,
                                     single_cls=True if config.nc > 1 else False)

        # 校验数据集标签类别和配置文件的一致性
        mlc = np.concatenate(train_dataset.labels, 0)[:, 0].max()  # max label class
        assert mlc < config.nc, 'Label class %g exceeds nc=%g. Correct your labels or your model.' % (mlc, config.nc)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=config.batch_size,
                                                 num_workers=0,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 collate_fn=train_dataset.collate_fn)

        test_dataset = LoadImageAndLabels(img_path=config.val_dataset,
                                          img_size=config.img_size,
                                          batch_size=config.batch_size,
                                          hyp=config.hyp,
                                          single_cls=True if config.nc > 1 else False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=config.batch_size,
                                                      num_workers=0,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      collate_fn=test_dataset.collate_fn)

        # 设置模型的参数
        config.hyp['cls'] *= config.nc / 1.
        model.nc = config.nc
        model.hyp = config.hyp
        model.gr = 1.0    # giou loss ratio (obj_loss = 1.0 or giou)
        model.class_weights = labels_to_class_weights(train_dataset.labels, config.nc).to(config.device)    # 附加类权重
        model.names = config.class_names

        # 每种类别出现的频率
        labels = np.concatenate(train_dataset.labels, 0)
        c = torch.tensor(labels[:, 0])    # 拿到类别列

        # 对模型做指数滑动平均
        ema = torch_utils.ModelEMA(model)

        # 开始训练
        t0 = time.time()
        n_burn = max(3 * len(train_dataloader), 1000)  # burn-in iterations, max(3 epochs, 1k iterations)
        maps = np.zeros(config.nc)    # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)    # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        print('Image sizes %g train, %g test' % (config.img_size, config.img_size))
        print('Starting training for %g epochs...' % config.epochs)
        for epoch in range(start_epoch, config.epochs):
            model.train()

            mloss = torch.zeros(4, device=config.device)  # mean losses
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(train_dataloader))
            for i, (imgs, targets, paths, _) in pbar:
                ni = i + len(train_dataloader) * epoch  # number integrated batches (since train start)
                imgs = imgs.to(config.device).float() / 255.0    # 图像的归一化

                # Burn-in
                if ni <= n_burn:
                    xi = [0, n_burn]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    accumulate = max(1, np.interp(ni, xi, [1, 64 / config.batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, config.hyp['momentum']])

                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(config.device), model)
                loss.backward()

                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)

                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, config.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)



















