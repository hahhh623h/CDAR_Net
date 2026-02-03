import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch.nn import init
#DWT and IDWT
##########################################################################
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

def idwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return idwt_init(x)

#Coordinate Attention
##########################################################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CAttention(nn.Module):
    def __init__(self, inp, reduction_ratio=4.):
        super(CAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(4, int(inp/reduction_ratio))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = x * a_w * a_h
        return out

#ARB
##########################################################################
class ARB(nn.Module):
    def __init__(self, dim, att_reduction_ratio=4., bias=False):
        super(ARB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            CAttention(dim, att_reduction_ratio),
        )

    def forward(self, x):
        return x + self.conv(x)


## Layer Norm
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

#GRB
##########################################################################
class GRB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2., bias=False):
        super(GRB, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.conv1 = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x0 = self.norm(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x0)
        x3 = F.gelu(x1) * x2
        out = self.out(x3)
        return x + out


#EFEM
##########################################################################
'''class EFEM(nn.Module):
    def __init__(self, dim=48, att_reduction_ratio=4., ffn_expansion_factor=2., bias=False):
        super(EFEM, self).__init__()

        self.att = ARB(dim, att_reduction_ratio, bias)
        self.ffn = GRB(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.att(x)
        x = self.ffn(x)
        return x'''


class EFEM(nn.Module):
    """Enhanced Feature Extraction Module with dynamic channel handling"""

    def __init__(self, dim, att_reduction_ratio=4, ffn_expansion_factor=2, bias=False):
        super(EFEM, self).__init__()
        # 确保输入通道数有效
        self.dim = int(dim)

        # 计算缩减后的维度，确保至少为1
        reduced_dim = max(1, dim // att_reduction_ratio)

        # 计算FFN隐藏层维度，确保至少为1
        hidden_dim = max(1, int(dim * ffn_expansion_factor))
        print(f"Creating EFEM with dim={self.dim}, reduced_dim={reduced_dim}, hidden_dim={hidden_dim}")
        # 通道注意力模块
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        # 空间注意力模块
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=bias),
            nn.Conv2d(reduced_dim, 1, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=bias)
        )

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通道注意力机制
        channel_att = self.channel_att(x)
        x_channel = x * channel_att

        # 空间注意力机制
        spatial_att = self.spatial_att(x_channel)
        x_spatial = x_channel * spatial_att

        # 前馈网络增强特征
        x_ffn = self.ffn(x_spatial)

        # 残差连接保留原始特征信息
        return x_ffn + self.conv(x_spatial)


#WaveCNN_CR
##########################################################################
class WaveCNN_CR(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dim=48, num_blocks=3, att_reduction_ratio=4.0, ffn_expansion_factor=2.0, bias=False):

        super(WaveCNN_CR, self).__init__()

        self.patch_embed = nn.Conv2d(input_nc, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.down0_1 = DWT()  ## From Level 0 to Level 1
        self.encoder_level1 = nn.Sequential(*[EFEM(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down1_2 = DWT()  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[EFEM(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down2_3 = DWT()  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[EFEM(dim*3, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.down3_4 = DWT()  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up4_3 = IDWT()  ## From Level 4 to Level 3
        self.decoder_level3 = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up3_2 = IDWT()  ## From Level 3 to Level 2
        self.decoder_level2 = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up2_1 = IDWT()  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[EFEM(dim*4, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.up1_0 = IDWT()  ## From Level 1 to Level 0  (NO 1x1 conv to reduce channels)
        self.refinement = nn.Sequential(*[EFEM(dim, att_reduction_ratio, ffn_expansion_factor, bias) for i in range(num_blocks)])

        self.output = nn.Conv2d(int(dim), output_nc, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_l, inp_enc_level1_h = self.down0_1(inp_enc_level1)

        out_enc_level1_h = self.encoder_level1(inp_enc_level1_h)
        inp_enc_level2_l, inp_enc_level2_h = self.down1_2(inp_enc_level1_l)

        out_enc_level2_h = self.encoder_level2(inp_enc_level2_h)
        inp_enc_level3_l, inp_enc_level3_h = self.down2_3(inp_enc_level2_l)

        out_enc_level3_h = self.encoder_level3(inp_enc_level3_h)
        inp_enc_level4_l, inp_enc_level4_h = self.down3_4(inp_enc_level3_l)

        inp_enc_level4 = torch.cat([inp_enc_level4_l, inp_enc_level4_h], 1)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3_l = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3_l, out_enc_level3_h], 1)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2_l = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2_l, out_enc_level2_h], 1)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1_l = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1_l, out_enc_level1_h], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.up1_0(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


import torch
import torch.nn as nn
import math


class se_block(nn.Module):
    def __init__(self, in_channels, ratio):
        super(se_block, self).__init__()
        middle_channels = in_channels // ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze phase
        squeezed = self.squeeze(x)
        print("Squeezed shape:", squeezed.shape)
        # Excitation phase
        weights = self.excitation(squeezed)
        print("Excitation weights shape:", weights.shape)
        # Re-calibration phase
        output = x * weights
        print("Output shape:", output.shape)
        return output

import torch.nn as nn
import torch
from torch.nn import functional as F


class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1

class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=1)
        # step 1: feature gating
        global_descriptors = torch.bmm(
            tmpA, attention_maps.permute(0, 2, 1)
        )  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ
#########################################
#########################################
# SKAttention
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import init


class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=group,
                                ),
                            ),
                            ("bn", nn.BatchNorm2d(channel)),
                            ("relu", nn.ReLU()),
                        ]
                    )
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V
########################################################################################################################################
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out
##########################################################################
import torch
import torch.nn as nn


class ECA(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

##########################################################################################################

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1

########################################################################################################################
import numpy as np
import torch
from torch import nn
from torch.nn import init


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, : c // 4] = x[:, : w - 1, :, : c // 4]
    x[:, : w - 1, :, c // 4 : c // 2] = x[:, 1:, :, c // 4 : c // 2]
    x[:, :, 1:, c // 2 : c * 3 // 4] = x[:, :, : h - 1, c // 2 : c * 3 // 4]
    x[:, :, : h - 1, 3 * c // 4 :] = x[:, :, 1:, 3 * c // 4 :]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, : c // 4] = x[:, :, : h - 1, : c // 4]
    x[:, :, : h - 1, c // 4 : c // 2] = x[:, :, 1:, c // 4 : c // 2]
    x[:, 1:, :, c // 2 : c * 3 // 4] = x[:, : w - 1, :, c // 2 : c * 3 // 4]
    x[:, : w - 1, :, 3 * c // 4 :] = x[:, 1:, :, 3 * c // 4 :]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c : c * 2])
        x3 = x[:, :, :, c * 2 :]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x
########################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B, H, W, device):
    # Create an infinite diagonal tensor on the specified device
    return (
        -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0)
        .unsqueeze(0)
        .repeat(B * W, 1, 1)
    )


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device
        self.to(device)
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = (
            proj_query.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
            .permute(0, 2, 1)
        )
        proj_query_W = (
            proj_query.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
            .permute(0, 2, 1)
        )
        proj_key = self.key_conv(x)
        proj_key_H = (
            proj_key.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_key_W = (
            proj_key.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        proj_value = self.value_conv(x)
        proj_value_H = (
            proj_value.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_value_W = (
            proj_value.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        energy_H = (
            (
                torch.bmm(proj_query_H, proj_key_H)
                + self.INF(m_batchsize, height, width, device)
            )
            .view(m_batchsize, width, height, height)
            .permute(0, 2, 1, 3)
        )
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(
            m_batchsize, height, width, width
        )
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = (
            concate[:, :, :, 0:height]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * width, height, height)
        )
        # print(concate)
        # print(att_H)
        att_W = (
            concate[:, :, :, height : height + width]
            .contiguous()
            .view(m_batchsize * height, width, width)
        )
        out_H = (
            torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
            .view(m_batchsize, width, -1, height)
            .permute(0, 2, 3, 1)
        )
        out_W = (
            torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
            .view(m_batchsize, height, -1, width)
            .permute(0, 2, 1, 3)
        )
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x
############################################################################################
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * 16, 4)
        layers = [nn.Sequential(
            nn.Conv2d(256, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d((1, 1)), #输出特征图的空间大小将被设置为 (1, 1)，这意味着无论你输入的特征图大小是多少，经过这个层后都会变成一个只包含一个元素的特征图（在空间维度上）
        )
        # input_channel = output_channel
        #
        # output_channel = 1280
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_channel, output_channel, bias=False),
        #     nn.BatchNorm1d(output_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        # 定义一个 1x1 卷积层，将通道数从 2048 减少到 256
        conv1x1 = nn.Conv2d(in_channels=960, out_channels=256, kernel_size=1)
        # 假设我们将使用 GPU（如果有的话）
        if torch.cuda.is_available():
            # 假设我们将使用 GPU（如果有的话）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 将输入张量和模型都移动到 GPU（如果有的话），或者留在 CPU 上
            x = x.to(device)
            conv1x1 = conv1x1.to(device)
            # 应用卷积层
            x = conv1x1(x)
        # 目标高度和宽度都是 64
        target_height = 32
        target_width = 32
        height = 2
        width = 2
        # 计算每个维度上需要填充的像素数
        pad_height = (target_height - height) // 2  # 向上取整，得到上面需要填充的像素数
        pad_bottom = target_height - height - pad_height  # 下面需要填充的像素数（可能与上面不同，如果总数是奇数）
        pad_width = (target_width - width) // 2  # 向上取整，得到左边需要填充的像素数
        pad_right = target_width - width - pad_width  # 右边需要填充的像素数（可能与左边不同，如果总数是奇数）
        x = F.pad(x, (pad_width, pad_right, pad_height, pad_bottom), mode='constant', value=0)

        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):  # 接受任意数量的关键字参数（**kwargs）
    """
    Constructs a GhostNet model
    ''''
        k：卷积核大小
        t：扩展比例（可能是 GhostNet 中使用的某种扩展策略）
        c：输出通道数
        SE：是否使用 SE（Squeeze-and-Excitation）块（0 表示不使用，1 表示使用）
        s：步长
        '''
        #
    """
    cfgs = [   # 定义了一个名为 cfgs 的列表，其中包含多个子列表，每个子列表表示 GhostNet 模型某一层的配置。每个子列表中的五个元素可能分别代表：

        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)


# if __name__=='__main__':
#     model = ghost_net()
#     model.train()
#     # model.eval()   # 使用 eval() 方法将模型设置为评估模式（关闭某些在训练时启用的功能，如 Dropout 和 BatchNorm 的训练模式）
#     print(model)
#     input = torch.randn(2,3,256,256)
#     y = model(input)
#     print(y)

