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
from pathlib import Path
import paddle
import paddle.nn
import paddle.io
import paddle.static
from paddle.distributed import fleet
from src.models import create_model
from src.helper_functions.helper_functions import *
import argparse
import os

parser = argparse.ArgumentParser(description='PaddlePaddle TResNet ImageNet Inference')

parser.add_argument('--train_mode', action='store_true', default=False)
parser.add_argument('--val_mode', action='store_true', default=False)
parser.add_argument('--infer_mode', action='store_true', default=False)

parser.add_argument('--params_dir', type=str, default='params')
parser.add_argument('--data_dir', type=str, default='train')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--model_name', type=str, default='tresnet_m')

parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--l2_decay', type=float, default=0.1)

parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--epoch_num', type=int, default=300)



def main():
    # parsing args
    args = parser.parse_args()
    # setup model
    print('creating model...')
    fleet.init(is_collective=True)
    model = create_model(args)
    # setup params
    params_dir = Path(args.params_dir)
    if not params_dir.exists():
        os.mkdir(args.params_dir)
    try:
        temp = os.listdir(args.params_dir)
        if args.input_size == 224:
            if args.model_name+'.pdparams' not in temp:
                print('no params...')
                choice = input('Do you want to create a new parameter file? [y/N]')
                if choice in ['Y', 'y']:
                    if args.input_size == 224:
                        paddle.save(model.state_dict(), args.params_dir+'/'+args.model_name+'.pdparams')
                    elif args.input_size == 448:
                        paddle.save(model.state_dict(), args.params_dir+'/'+args.model_name+'_448.pdparams')
                else:
                    print('done\n')
                    return
            model_params = paddle.load(args.params_dir+'/'+args.model_name+'.pdparams')
        elif args.input_size == 448:
            if args.model_name+'_448.pdparams' not in temp:
                print('no params...')
                return
            model_params = paddle.load(args.params_dir+'/'+args.model_name+'_448.pdparams')
        model.set_state_dict(model_params)
    except:
        print('params load error')
        return
    print('done\n')

    if args.train_mode == True:
        print('doing training...')
        train(args, model)
        print('done\n')
    elif args.val_mode == True:
        print('doing validation...')
        model.eval()
        data_load = get_load_dataset(args)
        prec1_f = validate(model, data_load)
        print("final top-1 validation accuracy: {:.2f}".format(prec1_f.avg))
        print('done\n')
    elif args.infer_mode == True:
        print('infer...')
        model.eval()
        tfms = get_val_tfms(args)
        dataset = forecastDataset(args, tfms)
        f = open('./result.csv', 'w')
        f.write('ImagePath, result\n')
        for idx in range(dataset.__len__()):
            img, path = dataset.__getitem__(idx)
            out = model(img)
            _, pred = out.topk(1, 1, True, True)
            print(f'${idx} [{path}]=>({int(pred[0])})')
            f.write(f'{path}, {int(pred[0])}\n')
        f.close()
        print('done\n')

if __name__ == '__main__':
    main()
