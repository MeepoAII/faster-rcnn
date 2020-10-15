from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

# from model.faster_rcnn import FasterRCNN
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
    del classfier[6]

    # delete 2 dropout layer
    if not opt.use_drop:
        del classfier[5]
        del classfier[2]

    classfier = nn.Sequential(*classfier)


    # freeze the first two stages of vgg16 without bp
    for layer in features[:10]:
        for p in layer.parameters():
            p.require_grad = False

    return nn.Sequential(*features), classfier


# Instantiation of feature extraction, classification part
# RPN network, and RoIHead network, respectively


# class FasterRCNNVGG16(FasterRCNN):
#     feat_stride = 16 # downsample 16x for output of conv5
#     def __init__(self, n_fg_class=20,
#                  ratios=[0.5, 1, 2],
#                  anchor_scales=[8, 16, 32]):
#
#         # before conv5_3 classifier
#         extractor, classifier = decom_vgg16()





















