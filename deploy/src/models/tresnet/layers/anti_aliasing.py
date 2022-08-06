#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class AntiAliasDownsampleLayer(nn.Layer):
    def __init__(self, filt_size: int = 3, stride: int = 2,channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)

class Downsample(nn.Layer):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert self.filt_size == 3
        a = paddle.to_tensor([1., 2., 1.])
        filt = (a[:, None] * a[None, :])
        filt = filt / paddle.sum(filt)
        self.filt =  filt[None, None, :, :].tile((self.channels, 1, 1, 1))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        x = F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])
        return x
