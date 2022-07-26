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
from src.models.tresnet.layers.ABN import ABN

class conv2d_ABN(nn.Layer):
    def __init__(self, ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
        super(conv2d_ABN, self).__init__()
        self.conv = nn.Conv2D(
            ni,
            nf,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias_attr=False
        )
        self.abn = ABN(
            num_features=nf,
            activation=activation,
            activation_param=activation_param
        )

    def forward(self, input):
        x = self.conv(input)
        x = self.abn(x)
        return x
