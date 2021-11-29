import cv2
import numpy as np
import colorsys
from PIL import Image, ImageDraw

hsv_tuples = [(x / 80, 1., 1.)
              for x in range(80)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
np.random.seed(10101)
np.random.shuffle(colors)
np.random.seed(None)

im0_pil = Image.fromarray(np.array([640, 480, 3]))
draw = ImageDraw.Draw(im0_pil)

c = 5
print(type(colors[int(c)]))
col = list(colors[int(c)])
print(type(col))

for i in range(2):
    draw.rectangle(
        [100 + i, 100 + i, 200 - i, 300 - i],
        outline=[5, 5, 5])

im0 = np.array(im0_pil)
cv2.imshow("im0", im0)
cv2.waitKey()