# TResNet: High Performance GPU-Dedicated Architecture
> From [linstcl.cn](http://www.linstcl.cn) [PaddlePaddle](https://www.paddlepaddle.org.cn/), [AI Studio](https://aistudio.baidu.com)@[Baidu](https://www.baidu.com)

[paperV2](https://arxiv.org/pdf/2003.13630.pdf)

TResNet的Paddle复现版

> Tal Ridnik, Hussam Lawen, Asaf Noy, Itamar Friedman, Emanuel Ben Baruch, Gilad Sharir<br/>
> DAMO Academy, Alibaba Group

**Abstract**

> Many deep learning models, developed in recent years, reach higher
> ImageNet accuracy than ResNet50, with fewer or comparable FLOPS count.
> While FLOPs are often seen as a proxy for network efficiency, when
> measuring actual GPU training and inference throughput, vanilla
> ResNet50 is usually significantly faster than its recent competitors,
> offering better throughput-accuracy trade-off. In this work, we
> introduce a series of architecture modifications that aim to boost
> neural networks' accuracy, while retaining their GPU training and
> inference efficiency. We first demonstrate and discuss the bottlenecks
> induced by FLOPs-optimizations. We then suggest alternative designs
> that better utilize GPU structure and assets. Finally, we introduce a
> new family of GPU-dedicated models, called TResNet, which achieve
> better accuracy and efficiency than previous ConvNets. Using a TResNet
> model, with similar GPU throughput to ResNet50, we reach 80.7\%
> top-1 accuracy on ImageNet. Our TResNet models also transfer well and
> achieve state-of-the-art accuracy on competitive datasets such as
> Stanford cars (96.0\%), CIFAR-10 (99.0\%), CIFAR-100 (91.5\%) and
> Oxford-Flowers (99.1\%). They also perform well on multi-label classification and object detection tasks.


## Main Article Results && Reproduce Article Scores
请参见[TResNet](https://github.com/Alibaba-MIIL/TResNet)

## Reproduce Article Scores
在paddlepaddle环境中运行：
```bash
python run.py \
--val_mode \
--params_dir=/model/path \
--data_dir=/path/to/val \
--model_name=tresnet_m \
--input_size=224
```

## Execute forecast
在paddlepaddle环境中运行：
```bash
python run.py \
--infer_mode \
--params_dir=/model/path \
--data_dir=/path/to/imgpath \
--model_name=tresnet_m \
--input_size=224
```

## TResNet Training
在paddlepaddle环境中运行：
```bash
python run.py \
--train_mode \
--params_dir=/model/path \
--data_dir=/path/to/train \
--model_name=tresnet_m \
--input_size=224 \
--batch_size=190 \
--epoch_num=300 \
--lr=0.2 \
--l2_decay=0.0001
```
<br>
也支持多卡训练：
```bash
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
```

## Citation

```
@misc{ridnik2020tresnet,
    title={TResNet: High Performance GPU-Dedicated Architecture},
    author={Tal Ridnik and Hussam Lawen and Asaf Noy and Itamar Friedman},
    year={2020},
    eprint={2003.13630},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact
联系复现作者(linstcl.cn)
Feel free to contact me if there are any questions or issues (Tal
Ridnik, tal.ridnik@alibaba-inc.com).
