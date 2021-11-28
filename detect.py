import os
import shutil
import colorsys
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw


import config

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

        # 设置字体，颜色
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * config.img_size + 0.5).astype('int32'))
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.





