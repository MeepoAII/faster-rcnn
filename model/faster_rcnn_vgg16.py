from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.faster_rcnn import FasterRCNN
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    # only take first 30 layer to extract feature
    features = list(model.features)[:30]

    classfier = model.classifier
    # put the classify section in a list
    classfier = list(classfier)

    # Delete the last full connection layer
