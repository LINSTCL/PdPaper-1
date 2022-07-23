import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.jit import to_static

class AntiAliasDownsampleLayer(nn.Layer):
    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_aa_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)

class DownsampleJIT(nn.Layer):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        super(DownsampleJIT, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert filt_size == 3
        assert stride == 2
        a = paddle.assign([1., 2., 1.])
        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / paddle.sum(filt)
        self.filt = filt[None, None, :, :].tile((self.channels, 1, 1, 1))

    def __call__(self, input: paddle.fluid.framework.Variable):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float() 
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=2, padding=0, groups=input.shape[1])


class Downsample(nn.Layer):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert self.filt_size == 3
        a = paddle.to_tensor([1., 2., 1.])
        filt = (a[:, None] * a[None, :])
        filt = filt / paddle.sum(filt)
        self.register_buffer('filt', paddle.tile(filt[None, None, :, :], [self.channels, 1, 1, 1]))
    
    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])
