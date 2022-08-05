import paddle
import paddle.nn as nn
import numpy as np

class ABN(nn.Layer):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.bn = paddle.nn.BatchNorm(
            num_channels=num_features,
            momentum=momentum,
            epsilon=eps,
            in_place=True
        )
        if activation == 'identity' or activation == None:
            self.a = paddle.nn.Identity()
        elif activation == 'leaky_relu':
            self.a = paddle.nn.LeakyReLU(activation_param)

    def forward(self, input):
        x = self.bn(input)
        x = self.a(x)
        return x

