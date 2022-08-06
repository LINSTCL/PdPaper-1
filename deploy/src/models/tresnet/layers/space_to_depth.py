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

class SpaceToDepthModule(nn.Layer):
    def __init__(self):
        super().__init__()
        self.op = SpaceToDepth()

    def forward(self, x):
        x = self.op(x)
        return x

class SpaceToDepth(nn.Layer):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, C, H // self.bs, self.bs, W // self.bs, self.bs])  # (N, C, H//bs, bs, W//bs, bs)
        x = x.transpose([0, 3, 5, 1, 2, 4])
        x = x.reshape([N, C * (self.bs ** 2), H // self.bs, W // self.bs])
        return x

class DepthToSpace(nn.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, self.bs, self.bs, C // (self.bs ** 2), H, W])  # (N, bs, bs, C//bs^2, H, W)
        x = x.transpose([0, 3, 4, 1, 5, 2]).clone()  # (N, C//bs^2, H, bs, W, bs)
        x = x.reshape([N, C // (self.bs ** 2), H * self.bs, W * self.bs])  # (N, C//bs^2, H * bs, W * bs)
        return x
