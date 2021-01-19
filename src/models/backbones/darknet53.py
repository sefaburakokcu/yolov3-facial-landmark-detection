''' The code is adapted from
https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py
'''
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out


class ResBlock(nn.Module):
    """The basic residual block used in YoloV3.
    Each ResBlock consists of two ConvLayers and the input is added to the final output.
    In YoloV3 paper, the first convLayer has half of the number of the filters as much as the second convLayer.
    The first convLayer has filter size of 1x1 and the second one has the filter size of 3x3.
    """

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is an even number.
        half_in_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels, half_in_channels, 1)
        self.conv2 = ConvLayer(half_in_channels, in_channels, 3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out


def make_conv_and_res_block(in_channels, out_channels, res_repeat):
    """In Darknet 53 backbone, there is usually one Conv Layer followed by some ResBlock.
    This function will make that.
    The Conv layers always have 3x3 filters with stride=2.
    The number of the filters in Conv layer is the same as the out channels of the ResBlock"""
    model = nn.Sequential()
    model.add_module('conv', ConvLayer(in_channels, out_channels, 3, stride=2))
    for idx in range(res_repeat):
        model.add_module('res{}'.format(idx), ResBlock(out_channels))
    return model


class DarkNet53(nn.Module):

    def __init__(self):
        super(DarkNet53, self).__init__()
        self.feat_channel = [1024, 512, 256]
        self.conv1 = ConvLayer(3, 32, 3)
        self.cr_block1 = make_conv_and_res_block(32, 64, 1)
        self.cr_block2 = make_conv_and_res_block(64, 128, 2)
        self.cr_block3 = make_conv_and_res_block(128, 256, 8)
        self.cr_block4 = make_conv_and_res_block(256, 512, 8)
        self.cr_block5 = make_conv_and_res_block(512, 1024, 4)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.cr_block1(tmp)
        tmp = self.cr_block2(tmp)
        out3 = self.cr_block3(tmp)
        out2 = self.cr_block4(out3)
        out1 = self.cr_block5(out2)

        return [out1, out2, out3]