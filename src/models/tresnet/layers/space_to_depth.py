import paddle
import paddle.nn as nn

class SpaceToDepthModule(nn.Layer):
    def __init__(self):
        super().__init__()
        self.op = SpaceToDepth()

    def forward(self, x):
        x = self.op(x)
        return x

class SpaceToDepth(nn.Layer):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, C, H // self.bs, self.bs, W // self.bs, self.bs])  # (N, C, H//bs, bs, W//bs, bs)
        x = x.transpose([0, 3, 5, 1, 2, 4])
        x = x.reshape([N, C * (self.bs ** 2), H // self.bs, W // self.bs])
        return x

class DepthToSpace(nn.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape([N, self.bs, self.bs, C // (self.bs ** 2), H, W])  # (N, bs, bs, C//bs^2, H, W)
        x = x.transpose([0, 3, 4, 1, 5, 2]).clone()  # (N, C//bs^2, H, bs, W, bs)
        x = x.reshape([N, C // (self.bs ** 2), H * self.bs, W * self.bs])  # (N, C//bs^2, H * bs, W * bs)
        return x
