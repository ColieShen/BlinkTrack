import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from src.model import Template
from model.loss import CrossEntropy, L1Truncated, UncertaintyLoss
from model.common import ConvBlock, ConvLSTMCell
from model.kalman_util import InvFuncMapping, ParabolaFuncMapping


class LSTMPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTMPredictor, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels, out_channels=64, n_convs=2, downsample=True
        )
        self.conv2 = ConvBlock(
            in_channels=64, out_channels=128, n_convs=2, downsample=True
        )
        self.convlstm0 = ConvLSTMCell(128, 128, 3)
        self.conv3 = ConvBlock(
            in_channels=128, out_channels=256, n_convs=2, downsample=True
        )
        self.conv4 = ConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=0,
            n_convs=1,
            downsample=False,
        )

        # Transformer Addition
        self.flatten = nn.Flatten()
        embed_dim = 256

        self.prev_x_res = None
        self.gates = nn.Linear(2 * embed_dim, embed_dim)

        # Attention Mask Transformer
        self.fusion_layer0 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
        )
        self.output_layers = nn.Sequential(nn.Linear(embed_dim, out_channels), nn.LeakyReLU(0.1))

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None

    def forward(self, x, attn_mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        if self.prev_x_res is None:
            self.prev_x_res = Variable(torch.zeros_like(x))

        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), 1))

        gate_weight = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), 1)))
        x = self.prev_x_res * gate_weight + x * (1 - gate_weight)

        self.prev_x_res = x

        x = self.output_layers(x)

        return x
    

class PyramidEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, recurrent=False, bottleneck=True):
        super(PyramidEncoder, self).__init__()

        self.conv_bottom_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            n_convs=2,
            kernel_size=1,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_00 = ConvBlock(
            in_channels=32,
            out_channels=64,
            n_convs=1,
            kernel_size=7,
            padding=3,
            downsample=False,
            stride=2,
        )
        self.conv_bottom_1 = ConvBlock(
            in_channels=64,
            out_channels=96,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_2 = ConvBlock(
            in_channels=96,
            out_channels=128,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=True,
        )
        self.conv_bottom_4 = ConvBlock(
            in_channels=256,
            out_channels=out_channels,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=False,
        )

        self.recurrent = recurrent
        if self.recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)

        self.conv_lateral_3 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_2 = nn.Conv2d(
            in_channels=128, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_1 = nn.Conv2d(
            in_channels=96, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_00 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_0 = nn.Conv2d(
            in_channels=32, out_channels=out_channels, kernel_size=1, bias=True
        )

        self.conv_dealias_3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_00 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_0 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.bottleneck = bottleneck
        if self.bottleneck:
            self.conv_bottleneck_out = nn.Sequential(
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    n_convs=1,
                    kernel_size=3,
                    padding=1,
                    downsample=False,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )

    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()

    def forward(self, x):
        """
        :param x:
        :return: (highest res feature map, lowest res feature map)
        """

        # Bottom-up pathway
        c0 = self.conv_bottom_0(x)  # 62x62
        c00 = self.conv_bottom_00(c0)   # 31x31
        c1 = self.conv_bottom_1(c00)  # 23x23
        c2 = self.conv_bottom_2(c1)  # 15x15
        c3 = self.conv_bottom_3(c2)  # 5x5
        c4 = self.conv_bottom_4(c3)  # 1x1

        # Top-down pathway (with lateral cnx and de-aliasing)
        p4 = c4
        p3 = self.conv_dealias_3(
            self.conv_lateral_3(c3)
            + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode="bilinear")
        )
        p2 = self.conv_dealias_2(
            self.conv_lateral_2(c2)
            + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode="bilinear")
        )
        p1 = self.conv_dealias_1(
            self.conv_lateral_1(c1)
            + F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode="bilinear")
        )
        p00 = self.conv_dealias_00(
            self.conv_lateral_00(c00)
            + F.interpolate(p1, (c00.shape[2], c00.shape[3]), mode="bilinear")
        )
        p0 = self.conv_dealias_0(
            self.conv_lateral_0(c0)
            + F.interpolate(p00, (c0.shape[2], c0.shape[3]), mode="bilinear")
        )

        if self.recurrent:
            p0 = self.conv_rnn(p0)

        if self.bottleneck:
            return self.conv_out(p0), self.conv_bottleneck_out(c4)
        else:
            return self.conv_out(p0)


def get_center_vector(x):
    # x.shape   B, C, P, P
    patch_size = x.shape[-1]
    if patch_size % 2 == 1:
        return x[..., patch_size // 2, patch_size // 2][..., None, None]
    else:
        center_idx = patch_size // 2
        center = x[..., center_idx-1:center_idx+1, center_idx-1:center_idx+1]   # B, C, 2, 2
        return torch.mean(center, dim=(-1, -2), keepdim=True)   # B, C, 1, 1


def get_center_patch(x, mid_patch_size):
    # x.shape   B, C, P, P
    patch_size = x.shape[-1]
    if (patch_size - mid_patch_size) % 2 == 1:
        x_tmp= get_center_patch(x, mid_patch_size+1)
        return (x_tmp[...,1:,1:] + x_tmp[...,1:,:-1] + x_tmp[...,:-1,1:] + x_tmp[...,:-1,:-1]) / 4
    else:
        edge_size = (patch_size - mid_patch_size) // 2
        return x[..., edge_size:patch_size-edge_size, edge_size:patch_size-edge_size]

class UncertaintyEncoder(nn.Module):
    def __init__(self, in_channels=257, out_channels=2, bias=False):
        super(UncertaintyEncoder, self).__init__()

        self.conv_bottom_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=128,
            n_convs=2,
            kernel_size=1,
            padding=0,
            downsample=False,
            bias=bias,
        )
        self.conv_bottom_1 = ConvBlock(
            in_channels=128,
            out_channels=64,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
            bias=bias,
        )
        self.conv_bottom_2 = ConvBlock(
            in_channels=64,
            out_channels=64,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
            bias=bias,
        )
        self.conv_bottom_3 = ConvBlock(
            in_channels=64,
            out_channels=64,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=True,
            bias=bias,
        )
        self.conv_bottom_4 = ConvBlock(
            in_channels=64,
            out_channels=128,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=False,
            bias=bias,
        )

        self.linear = nn.Linear(in_features=128, out_features=2, bias=False)
        # self.linear = nn.Linear(in_features=128, out_features=2, bias=True)

    def reset(self):
        pass

    def forward(self, x):

        # Bottom-up pathway
        x = self.conv_bottom_0(x)  # 31x31
        x = self.conv_bottom_1(x)  # 23x23
        x = self.conv_bottom_2(x)  # 15x15
        x = self.conv_bottom_3(x)  # 5x5
        x = self.conv_bottom_4(x)  # 1x1

        x = torch.flatten(x, start_dim=1)

        x = self.linear(x)

        return x


class BlinkTrack_Event(Template):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        feature_dim=384,
        patch_size=31,
        init_unrolls=1,
        input_channels=10,
        **kwargs,
    ):
        super(BlinkTrack_Event, self).__init__(
            representation=representation,
            max_unrolls=max_unrolls,
            init_unrolls=init_unrolls,
            n_vis=n_vis,
            patch_size=patch_size,
            **kwargs,
        )
        # Configuration
        self.grayscale_ref = True
        if not isinstance(input_channels, type(None)):
            self.channels_in_per_patch = input_channels

        # Architecture
        self.feature_dim = feature_dim
        self.redir_dim = 128

        self.reference_encoder = PyramidEncoder(1, self.feature_dim, bottleneck=False)
        self.target_encoder = PyramidEncoder(self.channels_in_per_patch, self.feature_dim, bottleneck=False)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Correlation3 had k=1, p=0
        self.reference_redir = nn.Conv2d(
            self.feature_dim, self.redir_dim, kernel_size=3, padding=1
        )
        self.target_redir = nn.Conv2d(
            self.feature_dim, self.redir_dim, kernel_size=3, padding=1
        )
        self.softmax = nn.Softmax(dim=2)

        self.lstm_predictor = LSTMPredictor(
            in_channels=(1 + 2 * self.redir_dim) * 2, out_channels=512
        )
        self.predictor = nn.Linear(in_features=512, out_features=2, bias=False)

        self.uncertainty_predictor = UncertaintyEncoder((1 + 2 * self.redir_dim) * 2, 2, bias=False)

        self.flatten = nn.Flatten()

        # Operational
        self.loss = L1Truncated(patch_size=patch_size*2)
        self.uncert_loss = CrossEntropy()
        # self.uncert_loss = UncertaintyLoss()
        # self.uncert2R = InvFuncMapping()
        self.uncert2R = ParabolaFuncMapping(thre=0.9)
        self.name = f"corr_{self.representation}"

        # Persistent Tensors
        self.f_ref, self.d_ref = None, None
        self.f_ref_mid, self.f_ref_squ = None, None

        self.correlation_maps = []
        self.inputs = []
        self.refs = []

        self.pyramid = True

        # freeze
        for name, child in self.named_children():
            if 'uncertainty_predictor' not in name:
                for param in child.parameters():
                    param.requires_grad = False

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc_out.weight)

    def reset(self, _):
        self.d_ref, self.f_ref = None, None
        self.f_ref_mid, self.f_ref_squ = None, None
        self.lstm_predictor.reset()

    def forward(self, x, xx, attn_mask=None):
        # Feature Extraction
        if isinstance(self.f_ref, type(None)):
            self.f_ref = self.reference_encoder(
                xx[:, self.channels_in_per_patch :, :, :]   
            )   # B, feature_dim, 62, 62
            self.d_ref = get_center_vector(self.f_ref)  # B, feature_dim, 1, 1
            self.f_ref = self.reference_redir(self.f_ref)   # B, redir_dim, 62, 62
            self.f_ref_mid = get_center_patch(self.f_ref, self.patch_size)  # B, redir_dim, 31, 31
            self.f_ref_squ = self.downsample(self.f_ref)    # B, redir_dim, 31, 31

        f0 = self.target_encoder(xx[:, : self.channels_in_per_patch, :, :])
        f0_mid = get_center_patch(f0, self.patch_size)
        f0_squ = self.downsample(f0)

        # Correlation and softmax
        f_corr_mid = (f0_mid * self.d_ref).sum(dim=1, keepdim=True)
        f_corr_mid = self.softmax(
            f_corr_mid.view(-1, 1, self.patch_size * self.patch_size)
        ).view(-1, 1, self.patch_size, self.patch_size)

        f_corr_squ = (f0_squ * self.d_ref).sum(dim=1, keepdim=True)
        f_corr_squ = self.softmax(
            f_corr_squ.view(-1, 1, self.patch_size * self.patch_size)
        ).view(-1, 1, self.patch_size, self.patch_size)

        # Feature Extraction
        f0 = self.target_redir(f0)
        f0_mid = get_center_patch(f0, self.patch_size)
        f0_squ = self.downsample(f0)

        # Feature re-direction
        f = torch.cat([f_corr_mid, f0_mid, self.f_ref_mid,
                       f_corr_squ, f0_squ, self.f_ref_squ], dim=1)
        
        uncertainty = self.uncertainty_predictor(f)

        f = self.lstm_predictor(f, attn_mask)
        coord = self.predictor(f)

        # coord = torch.zeros_like(uncertainty).to(uncertainty.device)   # only uncertainty train
        # coord = coord.detach()

        out = torch.cat([coord, uncertainty], dim=-1)

        return out
