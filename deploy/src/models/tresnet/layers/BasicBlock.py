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
import numpy as np
from src.models.tresnet.layers.conv2d_ABN import conv2d_ABN
from src.models.tresnet.layers.squeeze_and_excite import SEModule
import numpy as np

class BasicBlock(nn.Layer):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(
                    conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                    anti_alias_layer(channels=planes, filt_size=3, stride=2)
                )

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation='identity')
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(channels=planes * self.expansion, reduction_channels=reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = nn.functional.relu_(out)

        return out
