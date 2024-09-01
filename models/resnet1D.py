
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




# __all__ = [
#     "ResNet",
#     "ResNet18_Weights",
#     "ResNet34_Weights",
#     "ResNet50_Weights",
#     "ResNet101_Weights",
#     "ResNet152_Weights",
#     "ResNeXt50_32X4D_Weights",
#     "ResNeXt101_32X8D_Weights",
#     "ResNeXt101_64X4D_Weights",
#     "Wide_ResNet50_2_Weights",
#     "Wide_ResNet101_2_Weights",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "resnet152",
#     "resnext50_32x4d",
#     "resnext101_32x8d",
#     "resnext101_64x4d",
#     "wide_resnet50_2",
#     "wide_resnet101_2",
# ]

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


# In[4]:


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """5x5 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """1x1 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1_downsample(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding = 'valid', dilation: int = 1) -> nn.Conv1d:
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
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = norm_layer(len_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = norm_layer(len_feature)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        logger.debug(f"data shape after sub-conv1-3x3: {out.shape}")  # Log as debug
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        logger.debug(f"data shape after sub-conv2-3x3: {out.shape}")  # Log as debug
        out = self.ln2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            logger.debug(f"data shape after downsample: {identity.shape}")  # Log as debug

        out += identity
        out = self.relu(out)

        return out


# In[6]:


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)

# replace width to inplanes    -- yossi
        self.conv1 = conv1x1(inplanes, inplanes)
        
        self.ln1 = norm_layer(inplanes)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv2 = conv3x3(inplanes, inplanes, stride, groups, dilation)
        self.bn2 = norm_layer(inplanes)
        self.conv3 = conv1x1(inplanes, inplanes * self.expansion)
        self.bn3 = norm_layer(inplanes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        logger.debug(f"data shape after sub-conv1-1x1: {out.shape}")  # Log as debug
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        logger.debug(f"data shape after sub-conv2-3x3: {out.shape}")  # Log as debug
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        logger.debug(f"data shape after sub-conv3-1x1: {out.shape}")  # Log as debug
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# In[7]:


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
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
        self.inplanes = self.n_channels * 8     #   --yossi
        self.dilation = 1
        self.d_model = d_model
        self.len_feature = len_feature
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(self.n_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)    #  n_channels*8 = 152  --yossi
        
        self.ln1 = norm_layer(ceil(self.len_feature/2))    #   --yossi
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)    
        #   len_feature = len / 4   1500,   --yossi
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], ceil(self.len_feature/4))   #   channels*8 = 152, len_feature = len / 4   1500,   --yossi --yossi
        self.layer2 = self._make_layer(block, self.inplanes*2, layers[1], ceil(self.len_feature/8), stride=2, dilate=replace_stride_with_dilation[0])   # channels*16 = 304, len_feature = len / 8   750 --yossi
        self.layer3 = self._make_layer(block, self.inplanes*2, layers[2], ceil(self.len_feature/16), stride=2, dilate=replace_stride_with_dilation[1])    # channels*32 = 608, len_feature = len / 16   375  --yossi
        self.layer4 = self._make_layer(block, self.inplanes*2, layers[3], ceil(self.len_feature/32), stride=2, dilate=replace_stride_with_dilation[2])   # channels*64 = 1216, len_feature = len / 32, 188    --yossi
        self.avgpool = nn.AdaptiveAvgPool2d((self.n_channels, self.d_model))    # need to modify  --zys
        self.dropout2 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)    # --zys
        self.fc = nn.Linear(self.n_channels * self.d_model * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
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
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
            downsample = nn.Sequential(          #  --yossi
                conv1x1_downsample(self.inplanes, planes, stride, padding=0),
                norm_layer(len_feature),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, len_feature, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
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
        x = self.ln1(x)
        logger.debug(f"data shape after bn1: {x.shape}")  # Log as debug
        x = self.relu(x)
        x = self.maxpool(x)
        logger.debug(f"data shape after maxpool: {x.shape}")  # Log as debug

        logger.debug(f"layer1")  # Log as debug
        x = self.layer1(x)
#         logger.debug(f"data shape after layer1: {x.shape}")  # Log as debug
        x = self.dropout2(x)
        logger.debug(f"layer2")  # Log as debug
        x = self.layer2(x)
#         logger.debug(f"data shape after layer2: {x.shape}")
        x = self.dropout2(x)
        logger.debug(f"layer3")  # Log as debug
        x = self.layer3(x)
#         logger.debug(f"data shape after layer3: {x.shape}")
        x = self.dropout2(x)
        logger.debug(f"layer4")  # Log as debug
        x = self.layer4(x)
#         logger.debug(f"data shape after layer4: {x.shape}")
        x = self.dropout2(x)
        

        x = self.avgpool(x)
        logger.debug(f"data shape after avgpool: {x.shape}")
#         x = self.dropout5(x)
        
        x = torch.flatten(x, 1)
        logger.debug(f"data shape after flatten: {x.shape}")
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
#     weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


# In[9]:


# def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
def resnet18(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
#     weights = ResNet18_Weights.verify(weights)

#     return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
    return _resnet(BasicBlock, [2, 2, 2, 2], progress, **kwargs)


# In[10]:


def resnet34(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
#     weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], progress, **kwargs)


def resnet50(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
#     weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


def resnet101(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
#     weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], progress, **kwargs)


def resnet152(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """
#     weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], progress, **kwargs)


# In[19]:


# model = resnet34()


# In[20]:


# x = torch.randn(64, 19, 12000)
# output = model(x)

