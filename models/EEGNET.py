import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
from typing import List, Union
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import logging

import torch
from torch import Tensor, nn
from torch.types import Device, _size
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from configs.config import configs


logger = logging.getLogger(__name__)  # Use the current module's name
# logger.propagate = True



__all__ = [
    "EEGNet"
]


class EEGNet(nn.Module): 
    def __init__(self, signal_length: int = 12000, 
                 channel: int = 19, 
                 fs:int = 100,
                 num_class: int = 2,
                 F1:int = 8,
                 D:int = 3,
                ):
        super().__init__()
        num_input = 1
        kernel_size_1= (1,round(fs/2))
        kernel_size_2= (channel, 1)
        kernel_size_3= (1, round(fs/8))
        kernel_size_4= (1, 1)

        kernel_avgpool_1= (1,4)
        kernel_avgpool_2= (1,8)
        dropout_rate= 0.2

        ks0= int(round((kernel_size_1[0]-1)/2))
        ks1= int(round((kernel_size_1[1]-1)/2))
        kernel_padding_1= (ks0, ks1-1)
        ks0= int(round((kernel_size_3[0]-1)/2))
        ks1= int(round((kernel_size_3[1]-1)/2))
        kernel_padding_3= (ks0, ks1)
        F2= D*F1
        
        # layer 1
        self.conv2d = nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D*F1, kernel_size_2, groups= F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D*F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout_rate)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d( D*F1, D*F1, kernel_size_3,
                                                padding=kernel_padding_3, groups= D*F1)
        self.Separable_conv2D_point = nn.Conv2d(D*F1, F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(F2)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)
        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(F2*round(signal_length/32), num_class)
        self.Softmax = nn.Softmax(dim= 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        r"""Initiate parameters in the model."""
        
        for p in self.parameters():
            if p.dim() > 1:
#                 logger.debug(p.shape)
                nn.init.xavier_uniform_(p)
                    
        for m in self.modules():
#             print(m)
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print('Complete initiate parameters')
        
    def forward(self, x):
        x = x.transpose(-1,-2)
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x)) #.relu()
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)
        
        return y