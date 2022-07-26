import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def FastGlobalAvgPool2dJIT(x, flatten=False):
    in_size = x.shape
    if flatten:
        return x.reshape((in_size[0], in_size[1], -1)).mean(axis=2)
    else:
        return x.reshape([x.shape[0], x.shape[1], -1]).mean(-1).reshape([x.shape[0], x.shape[1], 1, 1])

class FastGlobalAvgPool2d(nn.Layer):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        in_size = x.shape
        if self.flatten:
            return x.reshape((in_size[0], in_size[1], -1)).mean(axis=2)
        else:
            return x.reshape([in_size[0], in_size[1], -1]).mean(-1).reshape([in_size[0], in_size[1], 1, 1])


