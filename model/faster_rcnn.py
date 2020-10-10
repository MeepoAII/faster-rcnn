from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
from torchvision.ops import nms

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


class FasterRCNN(nn.Module):

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_predict('evaluate')


    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
