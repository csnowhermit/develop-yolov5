import cv2
import numpy as np
import colorsys
from PIL import Image, ImageDraw
from models.yolo import Model

model = Model()
print(model)