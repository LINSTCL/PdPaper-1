import paddle
import paddle.nn as nn
import numpy as np
from src.models.tresnet.layers.conv2d_ABN import conv2d_ABN
from src.models.tresnet.layers.squeeze_and_excite import SEModule

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
                self.conv2 = nn.Sequential(
                    conv2d_ABN(
                        planes,
                        planes,
                        kernel_size=3,
                        stride=1,
                        activation="leaky_relu",
                        activation_param=1e-3
                    ),
                    anti_alias_layer(channels=planes, filt_size=3, stride=2)
                )

        self.conv3 = conv2d_ABN(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            activation="identity"
        )

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
