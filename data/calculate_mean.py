import os
import numpy as np
from util import read_image

path = r'/home/507/myfiles/data/voc/VOCdevkit/VOC2007/JPEGImages'


sum_R = 0
sum_G = 0
sum_B = 0

num = len(os.listdir(path))
print(f"num is {num}")

for fn in os.listdir(path):
    img_path = os.path.join(path, fn)
    img = read_image(img_path)
    # img /= 255
    sum_R += np.mean(img[0, :, :])
    sum_G += np.mean(img[1, :, :])
    sum_B += np.mean(img[2, :, :])

mean_R = sum_R / num
mean_G = sum_G / num
mean_B = sum_B / num

mean_array = np.array([mean_R, mean_G, mean_B]).reshape(3, 1, 1)
print(mean_array)
