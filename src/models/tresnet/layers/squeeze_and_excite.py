import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.models.tresnet.layers.avg_pool import FastGlobalAvgPool2d


class Flatten(nn.Layer):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(nn.Layer):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2D(channels, reduction_channels, kernel_size=1, padding=0, bias_attr=True)
        self.fc2 = nn.Conv2D(reduction_channels, channels, kernel_size=1, padding=0, bias_attr=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = nn.functional.relu_(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se
