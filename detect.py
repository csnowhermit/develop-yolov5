import os
import cv2
import time
import shutil
import colorsys
import torch
import numpy as np
from sys import platform
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

import config
from utils.dataset import LoadStream, LoadImages
from utils.util import non_max_suppression, xyxy2xywh, scale_coords, plot_one_box, formatTimestamp

'''
    推理过程
'''

if __name__ == '__main__':
    with torch.no_grad():
        model = torch.load(config.weight, map_location=config.device)['model']
        model.to(config.device).eval()

        if os.path.exists(config.output):
            shutil.rmtree(config.output)
        os.makedirs(config.output)

        # 只有cuda环境下才能使用半精度
        if config.half is True and config.use_gpu is True:
            model.half()

        # 将输入数据集做成DataLoader
        webcam = config.source == '0' or config.source.startswith('rtsp') or config.source.startswith('http') or config.source.endswith('.txt')
        if webcam:
            if config.use_gpu:
                torch.backends.cudnn.benchmark = True
            dataset = LoadStream(config.source)
        else:
            dataset = LoadImages(config.source)

        class_names = model.names if hasattr(model, 'names') else model.modules.names    # 总类别列表

        # 设置颜色
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # 开始推理
        t0 = time.time()
        for path, img, im0s, cap in dataset:
            img = torch.from_numpy(img).to(config.device)
            img = img.half() if config.half else img.float()    # 半精度计算
            img /= 255.0    # 归一化
            if img.ndimension() == 3:
                img = img.unsqueeze(0)    # 新增一个batch维度，做成BCHW的

            # 推理
            t1 = time.time()
            pred = model(img, augment=config.augment)[0]
            t2 = time.time()

            # nms
            if config.half:
                pred = pred.float()

            # Apply NMS。pred: list(tensor)
            pred = non_max_suppression(pred, conf_thres=config.conf_thres, iou_thres=config.iou_thres,
                                       fast=True, classes=None, agnostic=config.agnostic_nms)

            # 根据每张图片大小设置字体和画框的粗细
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * im0s.shape[0] + 0.5).astype('int32'))
            thickness = (im0s.shape[0] + im0s.shape[1]) // 300

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s    # im0:np.ndarray，这里im0s只有一张图片了
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(config.output) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # 放缩图像大小由img到原始shape
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, class_names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if config.save_img:
                            label = '%s %.2f' % (class_names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # 保存推理结果
                if config.save_img:
                    if dataset.mode == 'images':    # 目前只有输入图片需要保存
                        cv2.imwrite(save_path, im0)

        print('Done. (%.3fs)' % (time.time() - t0))



