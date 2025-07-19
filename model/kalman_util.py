import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvFuncMapping(nn.Module):
    # 0 - 1 to 0 - +inf
    # thre to 1

    def __init__(self, thre=0.9):
        super(InvFuncMapping, self).__init__()
        self.a = 1 / (1 - thre) - 1

    def forward(self, uncert):
        vis = F.softmax(uncert, dim=1)
        vis[:, 0] = 1 - vis[:, 1]
        vis[:, 1] = 1 - vis[:, 1]
        R = (1 / (1 - vis) - 1) / self.a
        R[R == 0] += 1e-8
        return R
    

class ParabolaFuncMapping(nn.Module):
    # 0 - 1 to 0 - up
    # thre to 1

    def __init__(self, thre=0.5, up=10):
        super(ParabolaFuncMapping, self).__init__()
        self.a = up
        self.b = math.log(1 / up, thre)

    def forward(self, uncert):
        vis = F.softmax(uncert, dim=-1)
        R = torch.zeros_like(vis).to(vis.device)
        R[:] = 1 - vis[..., 1:]
        R = self.a * torch.pow(R, self.b)
        R[R == 0] += 1e-8
        # print('event', R)
        return R
    

class CotrackerParabolaFuncMapping(nn.Module):
    # 0 - 1 to 0 - up
    # thre to 1

    def __init__(self, thre=0.1, up=2):
        super(CotrackerParabolaFuncMapping, self).__init__()
        self.a = up
        self.b = math.log(1 / up, thre)

    def forward(self, uncert):
        # uncert N, 1
        vis = torch.sigmoid(uncert)
        N = vis.shape[0]
        R = torch.zeros(N, 2).to(vis.device)
        R[:] = 1 - vis[..., 0:]
        R = self.a * torch.pow(R, self.b)
        R[R == 0] += 1e-8
        # print('rgb', R)
        return R
    

class CotrackerManualMapping(nn.Module):
    # 0 - 1 to 0 - up
    # thre to 1

    def __init__(self, thre=0.9, vis=1e-8, occ=2):
        super(CotrackerManualMapping, self).__init__()
        self.thre = thre
        self.vis = vis
        self.occ = occ

    def forward(self, uncert):
        # uncert N, 1
        vis = torch.sigmoid(uncert)
        N = vis.shape[0]
        R = torch.zeros(N, 2).to(vis.device)
        R[:] = vis[..., 0:]
        vis_mask = R >= self.thre
        R[~vis_mask] =  self.occ
        R[vis_mask] = self.vis
        return R
    
