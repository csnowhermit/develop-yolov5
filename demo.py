import os
import cv2
import glob
import numpy as np
import colorsys
from PIL import Image, ImageDraw
from models.yolo import Model

import torch
import config

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

path = "F:/dataset/balloon/images/train/"
f = glob.iglob(path + os.sep + '*.*')
img_files = [x.replace('/', os.sep) for x in f if
                  os.path.splitext(x)[-1].lower() in img_formats]

print(img_files)

label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in img_files]

print(label_files)