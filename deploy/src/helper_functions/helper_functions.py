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
import time, os
import paddle
import paddle.nn
import paddle.io
import paddle.vision
from paddle.distributed import fleet
import PIL.Image

def get_val_tfms(args):
    if args.input_size == 448: # squish
        val_tfms = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Resize((args.input_size, args.input_size))])
    else: # crop
        val_tfms = paddle.vision.transforms.Compose(
            [paddle.vision.transforms.Resize(int(args.input_size / args.val_zoom_factor)),
            paddle.vision.transforms.CenterCrop(args.input_size)])
    val_tfms.transforms.append(paddle.vision.transforms.ToTensor())
    return val_tfms
