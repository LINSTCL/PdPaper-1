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
import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL

def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model
