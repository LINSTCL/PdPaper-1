# TResNet: High Performance GPU-Dedicated Architecture
> From PaddlePaddle, AI Studio@Baidu

[paperV2](https://arxiv.org/pdf/2003.13630.pdf) |
[pretrained models](MODEL_ZOO.md)

Official PyTorch Implementation

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


## Main Article Results
#### TResNet Models
TResNet models accuracy and GPU throughput on ImageNet, compared to ResNet50. All measurements were done on Nvidia V100 GPU, with mixed precision. All models are trained on input resolution of 224.
<p align="center">
 <table>
  <tr>
    <th>Models</th>
    <th>Top Training Speed <br>(img/sec)</th>
    <th>Top Inference Speed<br>(img/sec)</th>
    <th>Max Train Batch Size</th>
    <th>Top-1 Acc.</th>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td><b>805</b></td>
    <td>2830</td>
    <td>288</td>
    <td>79.0</td>
  </tr>
  <tr>
    <td>EfficientNetB1</td>
    <td>440</td>
    <td>2740</td>
    <td>196</td>
    <td>79.2</td>
  </tr>
  <tr>
    <td>TResNet-M</td>
    <td>730</td>
    <td><b>2930</b></td>
    <td><b>512</b></td>
    <td>80.8</td>
  </tr>
  <tr>
    <td>TResNet-L</td>
    <td>345</td>
    <td>1390</td>
    <td>316</td>
    <td>81.5</td>
  </tr>
  <tr>
    <td>TResNet-XL</td>
    <td>250</td>
    <td>1060</td>
    <td>240</td>
    <td><b>82.0</b></td>
  </tr>
</table>
</p>

#### Comparison To Other Networks

Comparison of ResNet50 to top modern networks, with similar top-1 ImageNet accuracy.
 All measurements were done on Nvidia V100 GPU with mixed precision. For gaining optimal speeds, training and inference were measured on 90\% of maximal possible batch size.
 Except TResNet-M, all the models' ImageNet scores were taken from the [public repository](https://github.com/rwightman/pytorch-image-models), which specialized in providing top implementations for modern networks. Except EfficientNet-B1, which has input resolution of 240, all other models have input resolution of 224.
<p align="center">
<table class="tg">
  <tr>
    <th class="tg-c3ow">Model</th>
    <th class="tg-c3ow">Top Training Speed<br>(img/sec)</th>
    <th class="tg-c3ow">Top Inference Speed<br>(img/sec)</th>
    <th class="tg-c3ow">Top-1 Acc.</th>
    <th class="tg-c3ow">Flops[G]</th>
  </tr>
  <tr>
    <td class="tg-0pky">ResNet50</td>
   <td class="tg-c3ow"><b>805</b></td>
    <td class="tg-c3ow">2830</td>
    <td class="tg-c3ow">79.0</td>
    <td class="tg-c3ow">4.1</td>
  </tr>
  <tr>
    <td class="tg-0pky">ResNet50-D</td>
    <td class="tg-c3ow">600</td>
    <td class="tg-c3ow">2670</td>
    <td class="tg-c3ow">79.3</td>
    <td class="tg-c3ow">4.4</td>
  </tr>
  <tr>
    <td class="tg-0pky">ResNeXt50</td>
    <td class="tg-c3ow">490</td>
    <td class="tg-c3ow">1940</td>
    <td class="tg-c3ow">79.4</td>
    <td class="tg-c3ow">4.3</td>
  </tr>
  <tr>
    <td class="tg-0pky">EfficientNetB1</td>
    <td class="tg-c3ow">440</td>
    <td class="tg-c3ow">2740</td>
    <td class="tg-c3ow">79.2</td>
    <td class="tg-c3ow">0.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">SEResNeXt50</td>
    <td class="tg-c3ow">400</td>
    <td class="tg-c3ow">1770</td>
    <td class="tg-c3ow">79.9</td>
    <td class="tg-c3ow">4.3</td>
  </tr>
  <tr>
    <td class="tg-0pky">MixNet-L</td>
    <td class="tg-c3ow">400</td>
    <td class="tg-c3ow">1400</td>
    <td class="tg-c3ow">79.0</td>
    <td class="tg-c3ow">0.5</td>
  </tr>
  <tr>
    <td class="tg-0pky">TResNet-M</td>
    <td class="tg-c3ow">730</td>
   <td class="tg-c3ow"><b>2930</b></td>
    <td class="tg-c3ow"><b>80.8</b></td>
    <td class="tg-c3ow">5.5</td>
  </tr>
</table>
</p>

 <br/>
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./figures/table_4.png" align="center" width="400" ></td>
    <td class="tg-c3ow"><img src="./figures/table_5.png" align="center" width="400" ></td>
  </tr>
</table>
</p>


</p>

#### Transfer Learning SotA Results
Comparison of TResNet to state-of-the-art models on transfer learning datasets (only ImageNet-based transfer learning results). Models inference speed is measured on a mixed precision V100 GPU. Since no official implementation of  Gpipe was provided, its inference speed is unknown

<p align="center">
 <table style="border-collapse: collapse; border: none; border-spacing: 0px;">
	<tr>
		<td style="border-width: 1px; border-style: solid; border-color: black; padding-right: 3pt; padding-left: 3pt;">
			Dataset
		</td>
		<td style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Model
		</td>
		<td style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Top-1
			<br>
			Acc.
		</td>
		<td style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Speed
			<br>
			img/sec
		</td>
		<td style="border-right: 1px solid black; border-top: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			Input
		</td>
	</tr>
	<tr>
		<td rowspan="2" style="border-left: 1px solid black; border-right: 1px solid black; border-bottom: 2px double black; padding-right: 3pt; padding-left: 3pt;">
			CIFAR-10
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			Gpipe
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>99.0</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			-
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			480
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; padding-right: 3pt; padding-left: 3pt;">
			TResNet-XL
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>99.0</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>1060</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			224
		</td>
	</tr>
	<tr>
		<td rowspan="2" style="border-left: 1px solid black; border-right: 1px solid black; border-bottom: 2px double black; padding-right: 3pt; padding-left: 3pt;">
			CIFAR-100
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			EfficientNet-B7
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>91.7</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			70
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			600
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; padding-right: 3pt; padding-left: 3pt;">
			TResNet-XL
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			91.5
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>1060</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			224
		</td>
	</tr>
	<tr>
		<td rowspan="2" style="border-left: 1px solid black; border-right: 1px solid black; border-bottom: 2px double black; padding-right: 3pt; padding-left: 3pt;">
			 Stanford Cars
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			EfficientNet-B7
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			94.7
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			70
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			600
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; padding-right: 3pt; padding-left: 3pt;">
			TResNet-L
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>96.0</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>500</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 2px double black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			368
		</td>
	</tr>
	<tr>
		<td rowspan="2" style="border-left: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			 Oxford-Flowers
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			EfficientNet-B7
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			98.8
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			70
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			600
		</td>
	</tr>
	<tr>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; padding-right: 3pt; padding-left: 3pt;">
			TResNet-L
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>99.1</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			<b>500</b>
		</td>
		<td style="border-right: 1px solid black; border-bottom: 1px solid black; text-align: center; padding-right: 3pt; padding-left: 3pt;">
			368
		</td>
	</tr>
</table>
</p>


## Reproduce Article Scores
We provide code for reproducing the validation top-1 score of TResNet
models on ImageNet. First, download pretrained models from
[here](MODEL_ZOO.md).

Then, run the infer.py script. For example, for tresnet_m (input size 224)
run:
```bash
python -m infer.py \
--val_dir=/path/to/imagenet_val_folder \
--model_path=/model/path/to/tresnet_m.pth \
--model_name=tresnet_m
--input_size=224
```
## TResNet Training
Due to IP limitations, we do not provide the exact training code that
was used to obtain the article results.

However, TResNet is now an integral part of the popular [rwightman /
pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
repo. Using that repo, you can reach very similar results to the one
stated in the article.

For example, training tresnet_m on [rwightman /
pytorch-image-models](https://github.com/rwightman/pytorch-image-models) with
the command line:
```bash
python -u -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=1 --node_rank=0 ./train.py /data/imagenet/ \
-b=190 --lr=0.6 --model-ema --aa=rand-m9-mstd0.5-inc1 \
--num-gpu=8 -j=16 --amp \
--model=tresnet_m --epochs=300 --mixup=0.2 \
--sched='cosine' --reprob=0.4 --remode=pixel
```
gave accuracy of 80.5%. <br><br>


Also, during the merge request, we had interesting discussions and
insights regarding TResNet design. I am attaching a pdf version the
mentioned discussions. They can shed more light on TResNet design
considerations and directions for the future.

[TResNet discussion and insights](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/TResnet_discussion.pdf)

(taken with permission from
[here](https://github.com/rwightman/pytorch-image-models/issues/124))



## Tips For Working With Inplace-ABN
See
[INPLACE_ABN_TIPS](https://github.com/mrT23/TResNet/blob/master/INPLACE_ABN_TIPS.md).


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
Feel free to contact me if there are any questions or issues (Tal
Ridnik, tal.ridnik@alibaba-inc.com).
