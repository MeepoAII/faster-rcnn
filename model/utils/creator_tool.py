import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import loc2bbox

class ProposalCreator():
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size


    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        # 这里的loc和score是经过region_proposal_network中
        # 1x1卷积分类和回归得到的

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 将bbox 转化为近似groundtruth的anchor(rois)
        roi = loc2bbox(anchor, loc)

        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2), 0, img_size[0]]
        )
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2), 0, img_size[1]]
        )

        min_size = self.min_size * scale

        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        # 确保roi的长宽大于最小阈值
        keep = np.where((hs >= min_size) and (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.ravel().argsort()[::-1]

        if n_pre_nms > 0:
            order = order[:n_pre_nms]

        roi = roi[order, :]
        score = score[order]

        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh
        )
        if n_post_nms > 0:
            keep = keep[:n_post_nms]

        roi = roi[keep.cpu().numpy()]

        return roi





















