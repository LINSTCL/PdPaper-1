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

class ABN(nn.Layer):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.bn = paddle.nn.BatchNorm(
            num_channels=num_features,
            momentum=momentum,
            epsilon=eps,
            in_place=True
        )
        if activation == 'identity' or activation == None:
            self.a = paddle.nn.Identity()
        elif activation == 'leaky_relu':
            self.a = paddle.nn.LeakyReLU(activation_param)

    def forward(self, input):
        x = self.bn(input)
        x = self.a(x)
        return x

