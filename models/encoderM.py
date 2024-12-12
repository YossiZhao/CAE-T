
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from math import ceil

import torch
import torch.nn as nn
from torch import Tensor

# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface

# Init logging
import logging

logger = logging.getLogger(__name__)  # Use the current module's name
# logger.propagate = True



__all__ = [
    "AutoEncoder",
    "res_encoderS",
    "resnet18",
    "res_encoderM",
    "resnet34",
    "res_encoderL",
    "resnet50",
    "res_encoderXL",
    "resnet101",
    "resnet152",
    "MLP"
]


# In[4]:
class MLP(nn.Module):
    def __init__(self, in_planes: int, num_classes: int):
        super(MLP, self).__init__()
        print(in_planes, num_classes)
        self.fc1 = nn.Linear(in_planes, in_planes//8)  # First fully connected layer
        self.relu = nn.ReLU()          # Non-linearity
        self.fc2 = nn.Linear(in_planes//8, in_planes//32)   # Second fully connected layer
        self.fc3 = nn.Linear(in_planes//32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)                # Apply second layer
        x = self.relu(x)
        x = self.fc3(x)
        return x


def conv129(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=16,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1_downsample(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding = 'valid', dilation: int = 1) -> nn.Conv1d:
    """1x1 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


# In[5]:


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        len_feature: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv129(inplanes, planes, stride, groups=groups)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(len_feature)
        self.ln1 = norm_layer(len_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv129(planes, planes, groups=groups)
        self.avgpool_2 = nn.AdaptiveAvgPool1d(len_feature)
        self.ln2 = norm_layer(len_feature)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        logger.debug(f"data shape after sub-conv1: {out.shape}, groups={self.conv1.groups}")  # Log as debug
        out = self.avgpool_1(out)
        out = self.ln1(out)
        logger.debug(f"data shape after sub-ln1: {out.shape}")  # Log as debug
        out = self.relu(out)
        logger.debug(f"data shape after sub-relu1: {out.shape}")  # Log as debug

        out = self.conv2(out)
        logger.debug(f"data shape after sub-conv2: {out.shape}, groups={self.conv2.groups}")  # Log as debug
        out = self.avgpool_2(out)
        out = self.ln2(out)
        logger.debug(f"data shape after sub-ln2: {out.shape}")  # Log as debug

        if self.downsample is not None:
            identity = self.downsample(x)
            logger.debug(f"data shape after downsample: {identity.shape}, groups={self.downsample[0].groups}")  # Log as debug

        out += identity
        out = self.relu(out)
        logger.debug(f"data shape after residual relu+res: {out.shape}")  # Log as debug

        return out

class AutoEncoder(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_channels: int = 19,
        d_model: int = 256,
        len_feature: int = 12000     # --yossi
    ) -> None:
        super().__init__()
#         _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self._norm_layer = norm_layer
        self.n_channels = n_channels
        self.inplanes = self.n_channels * 8     #   from 19*8=152 to 19*64=1216  --yossi
        self.dilation = 1
        self.d_model = d_model   # 256  ---yossi
        self.len_feature = len_feature    # 12000
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups   # 19  --yossi
        self.base_width = width_per_group    # 64   --yossi
        self.conv1 = nn.Conv1d(self.n_channels, self.inplanes, kernel_size=64, groups=self.groups, stride=2, padding=3, bias=False)    #  n_channels*8 = 152  --yossi
        self.avgpool1d = nn.AdaptiveAvgPool1d(ceil(self.len_feature/2))
        self.ln1 = norm_layer(ceil(self.len_feature/2))    #   --yossi
        self.relu = nn.ReLU(inplace=True)
        self.avgpool_1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)    
        #   len_feature = len / 4   3000,   --yossi
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], ceil(self.len_feature/4))   #   channels*8 = 152, len_feature = len / 4   3000,   --yossi --yossi
        self.layer2 = self._make_layer(block, self.inplanes*2, layers[1], ceil(self.len_feature/8), stride=2, dilate=replace_stride_with_dilation[0])   # channels*16 = 304, len_feature = len / 8   750 --yossi
        self.layer3 = self._make_layer(block, self.inplanes*2, layers[2], ceil(self.len_feature/16), stride=2, dilate=replace_stride_with_dilation[1])    # channels*32 = 608, len_feature = len / 16   375  --yossi
        self.layer4 = self._make_layer(block, self.inplanes*2, layers[3], ceil(self.len_feature/32), stride=2, dilate=replace_stride_with_dilation[2])   # channels*64 = 1216, len_feature = len / 32, 188    --yossi
        self.layer4 = self._make_layer(block, self.inplanes*2, layers[4], ceil(self.len_feature/32), stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool_2 = nn.AdaptiveAvgPool2d((self.n_channels, self.d_model))    # need to modify  --zys
        self.dropout2 = nn.Dropout(0.2)
#         self.dropout5 = nn.Dropout(0.5)    # --zys
#         self.fc = nn.Linear(self.n_channels * self.d_model * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        len_feature: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(          #  --yossi
                conv1_downsample(self.inplanes, planes, stride, groups=self.groups, padding=0),
                norm_layer(len_feature),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, len_feature, stride, downsample, self.groups, self.base_width,
                previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    len_feature,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        logger.debug(f"data shape is: {x.shape}")  # Log as debug
        x = self.conv1(x)
        logger.debug(f"data shape after conv1: {x.shape}")  # Log as debug
        x = self.avgpool1d(x)
        x = self.ln1(x)
        logger.debug(f"data shape after ln1: {x.shape}")  # Log as debug
        x = self.relu(x)
        x = self.avgpool_1(x)
        logger.debug(f"data shape after avgpool: {x.shape}")  # Log as debug

        logger.debug(f"layer1")  # Log as debug
        x = self.layer1(x)
#         logger.debug(f"data shape after layer1: {x.shape}")  # Log as debug
        x = self.dropout2(x)
        logger.debug(f"layer2")  # Log as debug
        x = self.layer2(x)
#         logger.debug(f"data shape after layer2: {x.shape}")
#         x = self.dropout2(x)
        logger.debug(f"layer3")  # Log as debug
        x = self.layer3(x)
#         logger.debug(f"data shape after layer3: {x.shape}")
        logger.debug(f"layer4")  # Log as debug
        x = self.layer4(x)
#         logger.debug(f"data shape after layer4: {x.shape}")
#         x = self.dropout2(x)
        x = self.layer5(x)
#         logger.debug(f"data shape after layer3: {x.shape}")

        x = self.avgpool_2(x)
        logger.debug(f"data shape after avgpool: {x.shape}")
        
#         x = torch.flatten(x, 1)
#         logger.debug(f"data shape after flatten: {x.shape}")
        
#         x = self.mlp(x)
#         logger.debug(f"data shape after mlp: {x.shape}")
#         x = self.dropout5(x)
        
#         x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _autoencoder(
    block: Type[Union[BasicBlock]],
    layers: List[int],
    progress: bool,
    **kwargs: Any,
) -> AutoEncoder:

    model = AutoEncoder(block, layers, **kwargs)


    return model


# In[9]:

# 2min: 12000, 2^5-->375, avgpool-->256
def res_encoderS(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder:  

    return _autoencoder(BasicBlock, [1, 1, 1, 1], progress, **kwargs)

# 4min 24000, 2^6 -->375, avgpool-->256
def res_encoderM(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder: 

    return _autoencoder(BasicBlock, [2, 2, 2, 2, 2], progress, **kwargs)

# 8min 48000, 2^7 -->375, avgpool-->256
def res_encoderL(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder:

    return _autoencoder(BasicBlock, [3, 4, 6, 3, 2, 2], progress, **kwargs)

# 16min 96000, 2^8 -->375, avgpool-->256
def res_encoderXL(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder:

    return _autoencoder(Bottleneck, [3, 4, 8, 23, 8, 6, 3, 2], progress, **kwargs)


def res_encoderXXL(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder:

    return _autoencoder(Bottleneck, [3, 8, 36, 3], progress, **kwargs)




# In[20]:


# x = torch.randn(64, 19, 12000)
# output = model(x)

