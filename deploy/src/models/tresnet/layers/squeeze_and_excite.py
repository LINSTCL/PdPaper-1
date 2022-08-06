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
from src.models.tresnet.layers.avg_pool import FastGlobalAvgPool2d


class Flatten(nn.Layer):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(nn.Layer):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2D(channels, reduction_channels, kernel_size=1, padding=0, bias_attr=True)
        self.fc2 = nn.Conv2D(reduction_channels, channels, kernel_size=1, padding=0, bias_attr=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = nn.functional.relu_(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se
