from pathlib import Path
import paddle
import paddle.nn
from src.models import create_model
from src.helper_functions.helper_functions import *
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--params_path', type=str, default='params')
parser.add_argument('--train_dir', type=str, default='train')
parser.add_argument('--val_dir', type=str, default='val')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--train_mode', action='store_true', default=False)
parser.add_argument('--val_mode', action='store_true', default=False)

parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--momentum_rate', type=float, default=0.9)
parser.add_argument('--l2_decay', type=float, default=1e-4)

parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--epoch_num', type=int, default=10)



def main():
    # parsing args
    args = parser.parse_args()
    # setup model
    print('creating model...')
    model = create_model(args)
    params_path = Path(args.params_path)
    try:
        temp = params_path.resolve()
    except FileNotFoundError:
        # 不存在
        os.mkdir(args.params_path)
    else:
        # 存在
        try:
            temp = os.listdir(args.params_path)
            if args.model_name not in temp:
                print('creating now params...')
                temp = os.getcwd()
                os.chdir(args.params_path)
                paddle.save(model.state_dict(), args.model_name+'.pdparams')
                os.chdir(temp)
            temp = os.getcwd()
            os.chdir(args.params_path)
            model_params = paddle.load(args.model_name+'.pdparams')
            os.chdir(temp)
            model.set_state_dict(model_params)
        except:
            print('params load error')
            return
    print('done\n')

    if args.train_mode == True:
        train(args, model)

    if args.val_mode == True:
        print('doing validation...')
        model.eval()
        data_load = get_load_dataset(args)
        prec1_f = validate(model, data_load)
        print("final top-1 validation accuracy: {:.2f}".format(prec1_f.avg))

if __name__ == '__main__':
    main()
