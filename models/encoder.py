
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from math import ceil

import torch
import torch.nn as nn
from torch import Tensor


import logging

logger = logging.getLogger(__name__) 


__all__ = [
    "AutoEncoder",
    "res_encoderS",
    "res_encoderM",
    "res_encoderL",
]



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

        out = self.avgpool_1(out)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.avgpool_2(out)
        out = self.ln2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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
        len_feature: int = 12000    
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self._norm_layer = norm_layer
        self.n_channels = n_channels
        self.inplanes = self.n_channels * 4    
        self.dilation = 1
        self.d_model = d_model   
        self.len_feature = len_feature   
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False]*len(layers)
        self.groups = groups  
        self.base_width = width_per_group   
        self.conv1 = nn.Conv1d(self.n_channels, self.inplanes, kernel_size=64, groups=self.groups, stride=2, padding=3, bias=False)   
        self.avgpool1d = nn.AdaptiveAvgPool1d(ceil(self.len_feature/2))
        self.ln1 = norm_layer(ceil(self.len_feature/2))   
        self.relu = nn.ReLU(inplace=True)
        self.avgpool_1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)    
        
      
        self.layers = nn.ModuleList()
        for i, layer in enumerate(layers):
            if i == 0:
                self.layers.append(self._make_layer(block, self.inplanes, layer, ceil(self.len_feature/2**(i+2)), dilate=replace_stride_with_dilation[i]))
            else:
                self.layers.append(self._make_layer(block, self.inplanes*2, layer, ceil(self.len_feature/2**(i+2)), stride=2, dilate=replace_stride_with_dilation[i]))

        self.avgpool_2 = nn.AdaptiveAvgPool2d((self.groups, self.d_model))   
        self.dropout2 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)   


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0) 
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0) 

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

            downsample = nn.Sequential(         
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
      
        x = self.conv1(x)
        x = self.avgpool1d(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.avgpool_1(x)
                               
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.dropout2(x)

        x = self.avgpool_2(x)


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



# 2min: 12000, 2^5-->375, avgpool-->256
def res_encoderS(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder:  

    return _autoencoder(BasicBlock, [2, 1, 1, 1], progress, **kwargs)


def res_encoderM(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder: 

    return _autoencoder(BasicBlock, [2, 2, 2, 2, 2], progress, **kwargs)

def res_encoderL(*, weights = None, progress: bool = True, **kwargs: Any) -> AutoEncoder:

    return _autoencoder(BasicBlock, [3, 4, 6, 3, 2, 2], progress, **kwargs)




