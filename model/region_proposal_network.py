import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16,
                 proposal_create_params=dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios
        )
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_create_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)


    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)





def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)

    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]



def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal(mean, stddev)
        m.bias.data.zero_()