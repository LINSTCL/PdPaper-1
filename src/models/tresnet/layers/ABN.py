from typing import Optional
import paddle
import paddle.nn as nn
import numpy as np

class ABN(nn.Layer):
    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "num_features",
        "affine",
        "activation",
        "activation_param",
    ]
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    activation: str
    activation_param: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation
        self.activation_param = activation_param
        self.register_buffer("weight", paddle.ones(shape=[num_features]))
        self.register_buffer("bias", paddle.zeros(shape=[num_features]))
        self.register_buffer("running_mean", paddle.zeros(shape=[num_features]))
        self.register_buffer("running_var", paddle.ones(shape=[num_features]))
        self.num_batches_tracked = paddle.to_tensor([0], dtype=paddle.int64)

    def _get_momentum_and_training(self):
        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:
                    momentum = self.momentum

        if self.training:
            training = True
        else:
            training = (self.running_mean is None) and (self.running_var is None)

        return momentum, training

    def _get_running_stats(self):
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )
        return running_mean, running_var

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        x = nn.functional.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training,
            momentum,
            self.eps,
        )

        if self.activation == "relu":
            return nn.functional.relu_(x)
        elif self.activation == "leaky_relu":
            return nn.functional.leaky_relu(
                x, negative_slope=self.activation_param
            )
        elif self.activation == "elu":
            return nn.functional.elu_(x, alpha=self.activation_param)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation function {self.activation}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = paddle.to_tensor(0, dtype=paddle.int64)

        super(ABN, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}"
        if self.activation in ["leaky_relu", "elu"]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)
