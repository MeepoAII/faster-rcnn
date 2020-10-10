from data.dataset import *
from utils.config import Config
import cv2

opt = Config()
# print(opt.__dir__())

data = Dataset(opt)

img, bbox, label, scale = data.__getitem__(2)
print(img)
print(bbox)
print(label)
print(scale)
