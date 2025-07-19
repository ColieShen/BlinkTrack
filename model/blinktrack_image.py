import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

from model.kalman_util import ParabolaFuncMapping


def get_embedding(pos, C, cat_coords=True):
    B, N, D = pos.shape

    div_term = (torch.arange(0, C, 2, device=pos.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    pe_list = []

    for i in range(D):
        x = pos[:, :, i:i+1]
        pe_x = torch.zeros(B, N, C, device=pos.device, dtype=torch.float32)
        pe_x[:, :, 0::2] = torch.sin(x * div_term)
        pe_x[:, :, 1::2] = torch.cos(x * div_term)
        pe_list.append(pe_x)

    pe = torch.cat(pe_list, dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, pos], dim=2)  # B, N, C*3+3

    return pe


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def bilinear_sample2d(im, x, y, return_inbounds=False):
    # im -> [B, C, H, W]
    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device)*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B*H*W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
        w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(B, N)  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4):
        B, S, C, H, W = fmaps.shape
        # print('fmaps', fmaps.shape)
        self.S, self.C, self.H, self.W = S, C, H, W

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        # print('fmaps', fmaps.shape)

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels-1):
            fmaps_ = fmaps.reshape(B*S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)
            # print('fmaps', fmaps.shape)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert (D == 2)

        x0 = coords[:, 0, :, 0].round().clamp(0, self.W-1).long()
        y0 = coords[:, 0, :, 1].round().clamp(0, self.H-1).long()

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            _, _, _, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(B*S*N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(B*S*N, 1, H, W), coords_lvl)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert (C == self.C)
        assert (S == self.S)

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H*W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(input_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            # PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=8, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = 64
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim*2)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim*2)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim*2)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64,  stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128+96+64, output_dim, kernel_size=1)
        else:
            self.layer1 = self._make_layer(64,  stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)

            self.conv2 = nn.Conv2d(128+128+96+64, output_dim*2, kernel_size=3, padding=1, padding_mode='zeros')
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim*2, output_dim, kernel_size=1)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim_list = [item.shape[0] for item in x]
            x = torch.cat(x, dim=0)

        _, _, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)


        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(a, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a,b,c], dim=1))
        else:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(a, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            b = F.interpolate(b, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            c = F.interpolate(c, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            d = F.interpolate(d, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
            x = self.conv2(torch.cat([a,b,c,d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)
        
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, batch_dim_list, dim=0)

        return x

class DeltaBlock(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, corr_levels=4, corr_radius=3, depth=12):
        super(DeltaBlock, self).__init__()
        
        self.input_dim = input_dim

        kitchen_dim = (corr_levels * (2*corr_radius + 1)**2) + input_dim*2 + 64*2 + 2

        self.hidden_dim = hidden_dim

        self.to_delta = MLPMixer(
            input_dim=kitchen_dim,
            dim=512,
            output_dim=4,
            depth=depth,
        )
 
    def forward(self, fhid, target_ffeat, fcorr, flow):
        B, S, D = flow.shape
        assert(D==2)
        flow_sincos = get_embedding(flow, 64, cat_coords=True)
        x = torch.cat([fhid, target_ffeat, fcorr, flow_sincos], dim=2) # B, S, -1
        delta = self.to_delta(x)
        delta = delta.reshape(B, S, 4)
        return delta


class BlinkTrack_Image(nn.Module):
    def __init__(self, stride=8):
        super(BlinkTrack_Image, self).__init__()

        self.name = 'blinktrack'

        self.stride = stride    # squeeze factor
        self.hidden_dim = hdim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3
        self.iters = 3
        
        self.rgb_fnet = BasicEncoder(1, output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)
        
        self.rgb_delta_block = DeltaBlock(input_dim=self.latent_dim, hidden_dim=self.hidden_dim,
                                      corr_levels=self.corr_levels,
                                      corr_radius=self.corr_radius,
                                      depth=12)

        self.kalman_filter = None
        self.uncert2R = ParabolaFuncMapping(thre=0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_ref_view(self, ref_rgb, origin_xys):
        ref_coord = origin_xys.clone()/float(self.stride)

        rgb_ref_fmap = self.rgb_fnet(ref_rgb)
        rgb_ref_ffeat = bilinear_sample2d(rgb_ref_fmap, ref_coord[:,0,:,0], ref_coord[:,0,:,1]).permute(0, 2, 1) # B, N, C
        rgb_ref_ffeat = rgb_ref_ffeat[:, None]

        return rgb_ref_ffeat

    def predict_rgb(self, origin_xys, rgb, prev_state, ref_ffeat, ts, coords):
        B, _, N, D = origin_xys.shape
        rgb_ref_ffeat = ref_ffeat

        ref_coord = origin_xys.clone()/float(self.stride)

        if self.kalman_filter:
            prev_state = self.kalman_filter.detach(prev_state)
            flow_prev = prev_state['filter_state']['x'][:, :, :2, 0] / self.stride
            # flow_pred, prev_state, Q = self.kalman_filter.predict(prev_state, ts)
            # coords = ref_coord + flow_pred[:,None]
            coords = ref_coord + flow_prev[:, None]
        else:
            coords = coords.clone()/float(self.stride)

        rgb_fmaps = self.rgb_fnet(rgb)
        rgb_fmaps = rgb_fmaps[:, None]
        rgb_ref_ffeat_ = rgb_ref_ffeat.permute(0,2,1,3).reshape(B*N,1,self.latent_dim)
        rgb_fcorr_fn = CorrBlock(rgb_fmaps, num_levels=self.corr_levels, radius=self.corr_radius)
        rgb_fcorr_fn.corr(rgb_ref_ffeat)

        output_data = self._iter(coords, prev_state, ref_coord, rgb_fcorr_fn,
                                 rgb_fmaps, rgb_ref_ffeat_, 'rgb', ts)

        return output_data

    def _iter(self, coords, prev_state, ref_coord, fcorr_fn, ref_fmaps, ref_ffeat_, modal, ts):
        coord_predictions = []
        P_list = []

        for i in range(self.iters):
            coords = coords.detach()
            if self.kalman_filter:
                prev_state = self.kalman_filter.detach(prev_state)

            ref_fcorrs = fcorr_fn.sample(coords) # B, S, N, LRR
            B, _, N, LRR = ref_fcorrs.shape

            # for mixer, i want everything in the format B*N, S, C
            ref_fcorrs_ = ref_fcorrs.permute(0, 2, 1, 3).reshape(B*N, 1, LRR)
            flows_ = (coords - ref_coord).permute(0,2,1,3).reshape(B*N, 1, 2)

            target_ffeat = bilinear_sample2d(ref_fmaps[:,0], coords[:,0,:,0], coords[:,0,:,1]).permute(0, 2, 1) # B, N, C
            target_ffeat = target_ffeat[:, None]
            target_ffeat = target_ffeat.permute(0,2,1,3).reshape(B*N, 1, self.latent_dim)

            delta_block_fn = getattr(self, f'{modal}_delta_block')
            ref_ffeat_ = ref_ffeat_
            delta_all_ = delta_block_fn(ref_ffeat_, target_ffeat, ref_fcorrs_, flows_) # B*N, 1, C+2

            delta_coords_ = delta_all_[..., :2]
            delta_coords_ = delta_coords_.reshape(B, N, 1, 2).permute(0,2,1,3)
            coords = coords + delta_coords_

            if self.kalman_filter:
                # R_ = delta_all_[..., 2:]
                # R_ = F.softplus(R_).reshape(B, N, 2) + 1e-8
                # R_ = F.softplus(R_).reshape(B, N, 2) + 1e-3
                uncert = delta_all_[..., 2:].reshape(B, N, 2)    # B*N, 1, 2 -> B, N, 2
                R_ = self.uncert2R(uncert)
                R = torch.diag_embed(R_, dim1=-2, dim2=-1)  # B, N, 2, 2

                flow = (coords - ref_coord)[:, 0]   # frame 0 to t  B, N, 2
                flow, prev_state, S, P = self.kalman_filter.update(flow * self.stride, R, prev_state, ts)
                flow /= self.stride
                coords = ref_coord + flow[:, None]
                P_list.append(P[:,None])

            coord_predictions.append(coords * self.stride)

        return coord_predictions, prev_state, P_list
    

    def forward(self, func_name, *kwargs):
        assert func_name in ['init_ref_view', 'predict_rgb']

        if func_name == 'init_ref_view':
            return self.init_ref_view(*kwargs)

        elif func_name == 'predict_rgb':
            return self.predict_rgb(*kwargs)


if __name__ == '__main__':
    pass
