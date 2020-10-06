from __future__ import absolute_import
from __future__ import division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

def caffe_normalize(img):
    '''
    return appr -125~125 BGR
    :param img:
    :return:
    '''
    img = img[[2, 1, 0], :, :]
    img = img * 255
    mean = np.array([122.7717, 115.9645, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)

    return img

def pytorch_normalize(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                str=[0.229, 0.224, 0.225])

    img = normalize(t.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        pass



class Transform():

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip



