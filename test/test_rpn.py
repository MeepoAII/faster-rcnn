from model.faster_rcnn_vgg16 import decom_vgg16
from utils.config import opt
import torch
from PIL import Image

extractor, _ = decom_vgg16()
print(extractor)