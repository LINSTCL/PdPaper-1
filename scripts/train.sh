#! /bin/bash
python3 -m paddle.distributed.launch --gpus=0,1,2,3 run.py \
--train_mode \
--params_dir=/model/path \
--data_dir=/path/to/train \
--model_name=tresnet_m \
--input_size=224 \
--batch_size=190 \
--epoch_num=300 \
--lr=0.2 \
--l2_decay=0.0001
