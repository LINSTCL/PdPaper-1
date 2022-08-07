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

class FastGlobalAvgPool2d(nn.Layer):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        in_size = x.shape
        if self.flatten:
            return x.reshape((in_size[0], in_size[1], in_size[2]*in_size[3])).mean(axis=2)
        else:
            return x.reshape([in_size[0], in_size[1], in_size[2]*in_size[3]]).mean(-1).reshape([in_size[0], in_size[1], 1, 1])
