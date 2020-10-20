import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import loc2bbox

def _get_inside_index(anchor, H, W):
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

class AnchorTargetCreator():
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]






    def _create_label(self, inside_index, anchor, bbox):
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)




    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)




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
            roi[:, slice(0, 4, 2)], 0, img_size[0]
        )
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1]
        )

        min_size = self.min_size * scale

        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        # 确保roi的长宽大于最小阈值
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
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





















