import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn_trt import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn_trt import Model as ST_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()
        # m = torch.cat((torch.zeros(N, C, 1, V, M, device=x.device, dtype=x.dtype),
        #                 x[:, :, 1:] - x[:, :, :-1]), 2)

        m_diff = x[:, :, 1:] - x[:, :, :-1]
        m = F.pad(m_diff, (0, 0, 0, 0, 1, 0), "constant", 0)

        res = self.origin_stream(x) + self.motion_stream(m)
        return res