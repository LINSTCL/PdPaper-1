from numpy import pad
import paddle
import paddle.nn as nn
from functools import partial
from src.models.tresnet.layers.BasicBlock import BasicBlock
from src.models.tresnet.layers.Bottleneck import Bottleneck
from src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer
from src.models.tresnet.layers.avg_pool import FastGlobalAvgPool2d
from src.models.tresnet.layers.conv2d_ABN import conv2d_ABN
from src.models.tresnet.layers.space_to_depth import SpaceToDepthModule


class TResNet(nn.Layer):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0):
        super(TResNet, self).__init__()
        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = partial(AntiAliasDownsampleLayer)
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(
            BasicBlock,
            self.planes,
            layers[0],
            stride=1,
            use_se=True,
            anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(
            BasicBlock,
            self.planes * 2,
            layers[1],
            stride=2,
            use_se=True,
            anti_alias_layer=anti_alias_layer
        )  # 28x28
        layer3 = self._make_layer(
            Bottleneck,
            self.planes * 4,
            layers[2],
            stride=2,
            use_se=True,
            anti_alias_layer=anti_alias_layer
        )  # 14x14
        layer4 = self._make_layer(
            Bottleneck,
            self.planes * 8,
            layers[3],
            stride=2,
            use_se=False,
            anti_alias_layer=anti_alias_layer
        )  # 7x7

        # body
        self.body = nn.Sequential(
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)
        )

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(('global_pool_layer', global_pool_layer))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(('fc', fc))

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2D(kernel_size=2, stride=2, ceil_mode=True))
            layers += [
                conv2d_ABN(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    activation="identity"
                )
            ]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits

def TResnetM(model_params):
    """ Constructs a medium TResnet model.
    """
    num_classes = model_params['num_classes']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes)
    return model

