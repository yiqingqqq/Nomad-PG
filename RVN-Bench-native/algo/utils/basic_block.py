import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                #  style='pytorch',
                 with_cp=False,
                #  conv_cfg=None,
                #  norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""
        # print(x.shape)

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out