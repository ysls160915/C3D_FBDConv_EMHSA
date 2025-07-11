import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model.weight_init import constant_init, kaiming_init, normal_init
from mmengine.runner import load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmaction.registry import MODELS

class Attention3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1,
                 reduction=0.0625, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm3d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size**3, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(
            self.channel_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(
            self.filter_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        att = self.spatial_fc(x)
        att = att.view(x.size(0), 1, 1, 1,
                       self.kernel_size, self.kernel_size, self.kernel_size)
        return torch.sigmoid(att / self.temperature)

    def get_kernel_attention(self, x):
        att = self.kernel_fc(x)
        att = att.view(x.size(0), -1, 1, 1, 1, 1, 1)
        return F.softmax(att / self.temperature, dim=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = self.fc(x0)
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        return (self.func_channel(x0), self.func_filter(x0),
                self.func_spatial(x0), self.func_kernel(x0))

class FBDConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super().__init__()
        self.attn = Attention3D(in_planes, out_planes,
                                kernel_size, groups,
                                reduction, kernel_num)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes,
                        in_planes // groups,
                        kernel_size, kernel_size, kernel_size))
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self._init_weights()

    def _init_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        ca, fa, sa, ka = self.attn(x)
        B, C, D, H, W = x.shape
        x = x * ca
        x = x.view(1, B * C, D, H, W)
        agg = self.weight.unsqueeze(0) * sa * ka
        agg = agg.sum(dim=1)
        w = agg.view(B * self.out_planes,
                     self.in_planes // self.groups,
                     self.kernel_size,
                     self.kernel_size,
                     self.kernel_size)
        out = F.conv3d(x, weight=w, bias=None,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=self.groups * B)
        out = out.view(B, self.out_planes,
                       out.shape[-3], out.shape[-2], out.shape[-1])
        return out * fa

class EMHSA(nn.Module):
    """
    Multi-Head Self-Attention + Depthwise Separable Fusion
    """
    def __init__(self, spatial_dims, hidden_size,
                 num_heads=4, qkv_bias=False, attn_drop=0.1):
        super().__init__()
        self.spatial_dims = spatial_dims
        D, H, W = spatial_dims
        self.N = D * H * W
        self.hidden_size = hidden_size
        # always enable bias for correct tensor creation
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=True,
            batch_first=True
        )
        # zero out biases if qkv_bias is False
        if not qkv_bias:
            if self.mha.in_proj_bias is not None:
                nn.init.constant_(self.mha.in_proj_bias, 0)
            if hasattr(self.mha.out_proj, 'bias') and self.mha.out_proj.bias is not None:
                nn.init.constant_(self.mha.out_proj.bias, 0)

        self.dw = nn.Conv3d(hidden_size, hidden_size,
                            kernel_size=3, padding=1,
                            groups=hidden_size, bias=False)
        self.pw = nn.Conv3d(hidden_size, hidden_size,
                            kernel_size=1, bias=False)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        B, N, C = attn_out.shape
        vol = attn_out.permute(0, 2, 1).view(B, C, *self.spatial_dims)
        vol = self.dw(vol)
        vol = self.pw(vol)
        out = vol.view(B, C, -1).permute(0, 2, 1).contiguous()
        return out

@MODELS.register_module()
class C3D(nn.Module):
    def __init__(self, pretrained=None, style='pytorch',
                 conv_cfg=None, norm_cfg=None, act_cfg=None,
                 out_dim=8192, dropout_ratio=0.5,
                 init_std=0.005):
        super().__init__()
        conv_cfg = conv_cfg or dict(type='Conv3d')
        act_cfg = act_cfg or dict(type='ReLU')
        self.pretrained = pretrained
        c3d_param = dict(kernel_size=(3,3,3), padding=(1,1,1),
                         conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                         act_cfg=act_cfg)
        self.conv1a = ConvModule(3, 64, **c3d_param)
        self.pool1 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2a = ConvModule(64, 128, **c3d_param)
        self.pool2 = nn.MaxPool3d((2,2,2), (2,2,2))
        self.conv3a = ConvModule(128, 256, **c3d_param)
        self.conv3b = FBDConv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d((2,2,2), (2,2,2))
        self.conv4a = ConvModule(256, 512, **c3d_param)
        self.conv4b = FBDConv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d((2,2,2), (2,2,2))
        self.conv5a = ConvModule(512, 512, **c3d_param)
        self.conv5b = ConvModule(512, 512, **c3d_param)
        self.pool5 = nn.MaxPool3d((2,2,2), (2,2,2), padding=(0,1,1))
        self.emhsa = EMHSA(spatial_dims=(1,4,4), hidden_size=512, num_heads=4)
        self.fc6 = nn.Linear(out_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.init_std = init_std

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be str or None')

    def forward(self, x):
        x = self.conv1a(x); x = self.pool1(x)
        x = self.conv2a(x); x = self.pool2(x)
        x = self.conv3a(x); x = self.conv3b(x); x = self.pool3(x)
        x = self.conv4a(x); x = self.conv4b(x); x = self.pool4(x)
        x = self.conv5a(x); x = self.conv5b(x); x = self.pool5(x)
        B, C, D, H, W = x.shape
        seq = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        seq = self.emhsa(seq)
        x = seq.permute(0, 2, 1).view(B, C, D, H, W)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        return x
