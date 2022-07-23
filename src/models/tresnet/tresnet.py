from functools import partial

# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn
from collections import OrderedDict
from src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.avg_pool import FastGlobalAvgPool2d
from .layers.avg_pool import FastGlobalAvgPool2dJIT
from .layers.squeeze_and_excite import SEModule
from src.models.tresnet.layers.space_to_depth import SpaceToDepthModule


class conv2d_ABN(nn.Layer):
    def __init__(self, ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1, param_attr=None, bias_attr=None):
        super(conv2d_ABN, self).__init__()
        self.activation = activation
        self.activation_param = activation_param
        self.param_attr=param_attr
        self.bias_attr=bias_attr
        self.conv = nn.Conv2D(
            ni, 
            nf, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size // 2, 
            groups=groups,
            bias_attr=False
        )

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = paddle.fluid.layers.inplace_abn(
            x, 
            act=self.activation, 
            act_alpha=self.activation_param,
            param_attr=self.param_attr,
            bias_attr=self.bias_attr
        )
        return x



class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation=None, param_attr=paddle.fluid.initializer.Constant(0))
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = nn.functional.relu_(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation=None, param_attr=paddle.fluid.initializer.Constant(0))

        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = nn.functional.relu_(out)

        return out


class TResNet(nn.Layer):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, remove_aa_jit=False):
        super(TResNet, self).__init__()

        # JIT layers
        if not remove_aa_jit:
            space_to_depth = SpaceToDepthModule(remove_model_jit=False)## OK
            anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=False)## OK
            global_pool_layer = FastGlobalAvgPool2dJIT(flatten=True)## OK
        else:
            space_to_depth = SpaceToDepthModule(remove_model_jit=True)## OK
            anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=True)## OK
            global_pool_layer = FastGlobalAvgPool2d(flatten=True)## OK

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(
            in_chans * 16, 
            self.planes, 
            stride=1, 
            kernel_size=3,
            param_attr=paddle.fluid.initializer.Constant(1),
            bias_attr=paddle.fluid.initializer.Constant(0)
        )## OK
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)
        )## OK

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(('global_pool_layer', global_pool_layer))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(('fc', fc))

        # model initilization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2D):
        #         m.weight_attr = nn.initializer.KaimingNormal()
        #     elif isinstance(m, nn.BatchNorm2D):
        #         m.weight_attr = nn.initializer.Constant(1)
        #         m.bias_attr = nn.initializer.Constant(0)
        #     elif isinstance(m, nn.Linear):
        #         m.weight_attr = 

        # residual connections special initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2D(kernel_size=2, stride=2, ceil_mode=True, exclusive=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation=None)]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


def TResnetM(model_params):
    """ Constructs a medium TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    remove_aa_jit = model_params['remove_aa_jit']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans,
                    remove_aa_jit=remove_aa_jit)
    return model

