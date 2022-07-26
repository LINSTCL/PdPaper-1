import paddle
import paddle.nn as nn

class TResNet():

    def __init__(self, layers, num_classes=1000):
        super(TResNet, self).__init__()
        self.layers = layers
        self.fc_w = paddle.static.create_parameter(
            [2048,num_classes], 
            dtype='float32', 
            default_initializer=nn.initializer.Normal(0, 0.01)
        )
        self.fc_b = paddle.static.create_parameter([num_classes], dtype='float32')

    def FastGlobalAvgPool2dJIT(self, x, flatten=False):
        in_size = x.shape
        if flatten:
            return x.reshape((in_size[0], in_size[1], -1)).mean(axis=2)
        else:
            return x.reshape([x.shape[0], x.shape[1], -1]).mean(-1).reshape([x.shape[0], x.shape[1], 1, 1])

    def SpaceToDepthJit(self, x):
         # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.shape
        x = x.reshape([N, C, H // 4, 4, W // 4, 4])  # (N, C, H//bs, bs, W//bs, bs)
        x = x.transpose([0, 3, 5, 1, 2, 4]).clone()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.reshape([N, C * 16, H // 4, W // 4])  # (N, C*bs^2, H//bs, W//bs)
        return x

    def conv2d_ABN(
        self, 
        x, 
        nf, 
        stride, 
        activation="leaky_relu", 
        kernel_size=3, 
        activation_param=1e-2, 
        groups=1, 
        conv_param_attr=nn.initializer.KaimingNormal(),
        abn_param_attr=nn.initializer.Constant(1),
        abn_bias_attr=nn.initializer.Constant(0)
    ):
        x = paddle.static.nn.conv2d(
            x, 
            nf, 
            filter_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            param_attr=conv_param_attr,
            bias_attr=False
        )
        x = paddle.fluid.layers.inplace_abn(
            x, 
            act=activation, 
            act_alpha=activation_param,
            param_attr=abn_param_attr,
            bias_attr=abn_bias_attr
        )
        return x

    def anti_alias_layer(self, x, filt_size: int = 3, stride: int = 2, channels: int = 0):
        assert filt_size == 3
        assert stride == 2
        a = paddle.assign([1., 2., 1.])
        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / paddle.sum(filt)
        filt = filt[None, None, :, :].tile((channels, 1, 1, 1))
        if x.dtype != filt.dtype:
            filt = filt.float() 
        x = nn.functional.pad(x, (1, 1, 1, 1), 'reflect')
        x = nn.functional.conv2d(x, filt, stride=2, padding=0, groups=x.shape[1])
        return x

    def SEModule(self, x, channels, reduction_channels, inplace=True):
        x_se = self.FastGlobalAvgPool2dJIT(x)
        x_se2 = paddle.static.nn.conv2d(x_se, reduction_channels, 1, padding=0, bias_attr=True)
        if inplace:
            x_se2 = nn.functional.relu_(x_se2)
        else:
            x_se2 = nn.functional.relu(x_se2)
        x_se = paddle.static.nn.conv2d(x_se2, channels, 1, padding=0, bias_attr=True)
        x_se = nn.functional.sigmoid(x_se)
        return x * x_se

    def BasicBlock(
        self, 
        x,
        planes, 
        stride=1, 
        downsample1=None,
        downsample2=None, 
        use_se=True
    ):
        residual = None
        if downsample1 is not None:
            residual = downsample1(x, kernel_size=2, stride=2, ceil_mode=True, exclusive=False)
            if downsample2 is not None:
                residual = downsample2(residual, planes, kernel_size=1, stride=1, activation="identity")
        else:
            residual = x
        if stride == 1:
            x = self.conv2d_ABN(x, planes, stride=1, activation_param=1e-3)
        else:
            x = self.conv2d_ABN(x, planes, stride=1, activation_param=1e-3)
            x = self.anti_alias_layer(x, channels=planes)
        x = self.conv2d_ABN(x, planes, stride=1, activation='identity')
        if use_se: 
            reduce_layer_planes = max(planes // 4, 64)
            x = self.SEModule(x, planes, reduce_layer_planes)
        x += residual
        x = nn.functional.relu_(x)
        return x

    def Bottleneck(
        self, 
        x,
        planes, 
        stride=1, 
        downsample1=None,
        downsample2=None, 
        use_se=True
    ):
        residual = None
        if downsample1 is not None:
            residual = downsample1(x, kernel_size=2, stride=2, ceil_mode=True, exclusive=False)
            if downsample2 is not None:
                residual = downsample2(residual, planes*4, kernel_size=1, stride=1, activation="identity")
        else:
            residual = x
        x = self.conv2d_ABN(x, planes, kernel_size=1, stride=1, activation_param=1e-3)
        if stride == 1:
            x = self.conv2d_ABN(x, planes, kernel_size=3, stride=1, activation_param=1e-3)
        else:
            x = self.conv2d_ABN(x, planes, kernel_size=3, stride=1, activation_param=1e-3)
            x = self.anti_alias_layer(x, channels=planes)
        if use_se: 
            reduce_layer_planes = max(planes * 4 // 8, 64)
            x = self.SEModule(x, planes, reduce_layer_planes)
        x = self.conv2d_ABN(x, planes*4, kernel_size=1, stride=1, activation='identity')
        x += residual
        x = nn.functional.relu_(x)
        return x
    
    def conv1(self, x):
        x = self.conv2d_ABN(
            x, 
            64, 
            stride=1, 
            kernel_size=3, 
            conv_param_attr=nn.initializer.KaimingNormal()
        )
        return x

    def layer1(self, x):
        x = self.BasicBlock(x, 64, 1, use_se=True)
        for i in range(1, self.layers[0]):
            x = self.BasicBlock(x, 64, use_se=True)
        return x

    def layer2(self, x):
        downsample1 = nn.functional.avg_pool2d
        downsample2 = self.conv2d_ABN
        x = self.BasicBlock(x, 64*2, 2, use_se=True, downsample1=downsample1, downsample2=downsample2)
        for i in range(1, self.layers[1]):
            x = self.BasicBlock(x, 64*2, use_se=True)
        return x

    def layer3(self, x):
        downsample1 = nn.functional.avg_pool2d
        downsample2 = self.conv2d_ABN
        x = self.Bottleneck(x, 64*4, 2, use_se=True, downsample1=downsample1, downsample2=downsample2)
        for i in range(1, self.layers[2]):
            x = self.Bottleneck(x, 64*4, use_se=True)
        return x

    def layer4(self, x):
        downsample1 = nn.functional.avg_pool2d
        downsample2 = self.conv2d_ABN
        x = self.Bottleneck(x, 64*8, 2, use_se=False, downsample1=downsample1, downsample2=downsample2)
        for i in range(1, self.layers[3]):
            x = self.Bottleneck(x, 64*8, use_se=False)
        return x

    def net(self, x):
        x = self.SpaceToDepthJit(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.FastGlobalAvgPool2dJIT(x, True)
        x = nn.functional.linear(x, weight=self.fc_w, bias=self.fc_b)
        # out = self.head(x)
        return x


def TResnetM(model_params):
    """ Constructs a medium TResnet model.
    """
    num_classes = model_params['num_classes']
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes)
    return model

