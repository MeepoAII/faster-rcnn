from model.faster_rcnn_vgg16 import decom_vgg16
from model.region_proposal_network import RegionProposalNetwork
from data.dataset import Transform
from torchvision import datasets, transforms
import sys
sys.path.append("..")
from utils import config
opt = config.Config()

import torch
from PIL import Image
from matplotlib import pyplot as plt

image = Image.open("1.jpg").convert('RGB')
plt.imshow(image)
plt.show()

transform = transforms.Compose([transforms.ToTensor()])
img = transform(image)

# add batch dimension
img = img.unsqueeze(0)

# print(img.shape)
extractor, classifier = decom_vgg16()

features = extractor(img)
print(features.shape)
# one_dim_feature = out[:, 1, :, :]
#
# test = one_dim_feature.detach().numpy()
#
# print(one_dim_feature.shape)


ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]
feat_stride = 16    # vgg 是下采样16倍

rpn = RegionProposalNetwork(
    512, 512,
    ratios=ratios,
    anchor_scales=anchor_scales,
    feat_stride=feat_stride
)

# print(rpn)

_, _, H, W = img.shape
img_size = (H, W)
# use RPN
rpn_locs, rpn_scores, rois, roi_indices, anchor =\
    rpn(features, img_size)
print(rois.shape)