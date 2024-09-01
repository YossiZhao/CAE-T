import math
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init

### Define operation in auto-encoder
class Mat_mul(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #         self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return input @ self.weight + self.bias


#         return torch.mul(input, self.weight, self.bias)

### Define auto-encoder
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Mat_mul(input_size, hidden_size),
            nn.ReLU()
        )
        self.encoder_2 = nn.Sequential(
            Mat_mul(int(input_size / 2), hidden_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            Mat_mul(hidden_size, input_size),
            nn.ReLU()
        )
        self.decoder_2 = nn.Sequential(
            Mat_mul(int(input_size / 2), input_size),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        #         z = self.encoder_2(z)
        x_hat = self.decoder(z)
        #         x_hat = self.decoder_2(x_hat)
        return z