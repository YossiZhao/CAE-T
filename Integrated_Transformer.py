#!/usr/bin/env python
# coding: utf-8


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


# import torchonn as onn
# from torchonn.models import ONNBaseModel
# from torchonn.op.mzi_op import project_matrix_to_unitary


import torch
from torch import Tensor, nn
from torch.types import Device, _size
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchonn.layers import MZILinear
# from torchonn.models import ONNBaseModel
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from configs.config import configs

from models.resnet1D import *
from models.transformer_encoder import transformer_classifier

# ### Initilization


# Init logging


# logger = logging.getLogger(__name__)  # Use the current module's name
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# acc_example = 0.95  # Replace with your actual accuracy calculation
# logger.info(f"test accuracy: {acc_example}")  # Log as info
# logger.debug("Current accuracy: %.2f", accuracy)  # Log as info



# ### Dataset

class customDataset(Dataset):
    def __init__(self, data_dir:str, label_dir:str, label_dict:dict, transform=None):
#         self.annotations = pd.read_csv(label_dir)
        self.data_dir = data_dir   # './data/origin_csv/train'
        self.label_dir = label_dir  # './data/train_label.csv'
        self.transform = transform   
        self.files = os.listdir(self.data_dir)  # train_dataset.files[24580]='aaaaanrb_s002_t000_7.csv', abnormal
                                                # train_dataset.files[2]='aaaaaetn_s002_t000_6.csv', normal
        self.annotations = pd.read_csv(self.label_dir)   # (28434, 2)   aaaaaool_s004_t001_1.csv	abnormal
        self.label_dict = label_dict     #   {'normal': 0, 'abnormal': 1}
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.files[index])
        data = pd.read_csv(data_path)
        data = torch.tensor(data.values, dtype=torch.float32)
        file_name = self.files[index]
        label = self.annotations.loc[(self.annotations['csv_file']==file_name)]
        label = label.iloc[0,1]
        label_tensor = torch.tensor(int(self.label_dict[label]))
        
        if self.transform:
            data = self.transform(data)
            
        return (data.t(), label_tensor, file_name)




# label_dic = {'normal':0, 'abnormal':1}

# transform = transforms.Compose([
#     transforms.MinMaxScaler(feature_range=(0, 1)),
#     transforms.ToTensor(),
# ])
# combined_dataset = ConcatDataset([train_dataset, eval_dataset])


# ### Define model
### Define transformer_classifier
# class eeg_classifier(nn.Module):
#     def __init__(self, input_size:int, n_channels:int, model_hyp:dict, classes:int):
#         super(transformer_classifier, self).__init__()
#         self.ae = AutoEncoder(input_size=input_size, hidden_size=model_hyp['d_model'])
#         self.classifier = transformer_encoder(model_hyp:dict, n_channels:int, classes:int)
        
#     def forward(self, x):
#         z = self.ae(x)
#         logger.debug(f"au output size: %{z.shape}")
#         y = self.classifier(z)
#         logger.debug(f"transformer output size: %{z.shape}")
#         return y


### Define classifier
class model(nn.Module):
    def __init__(self, input_size: int, n_channels: int, model_hyp: dict, classes: int):
        super(model, self).__init__()
        self.ae = resnet18()
        self.transformer_encoder = transformer_classifier(input_size, n_channels, model_hyp, classes)

    def forward(self, x):
        z = self.ae(x)
        z = self.transformer_encoder(z)
        return z


### Learning rate update policy
def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if max_iter == 0:
        raise Exception("MAX ITERATION CANNOT BE ZERO!")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    logger.info(f'lr=: {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    

def train(Configs:dict):
    train_data_dir = Configs['dataset']['train_data_dir']
    train_label_dir = Configs['dataset']['train_label_dir']

    val_data_dir = Configs['dataset']['val_data_dir']
    val_label_dir = Configs['dataset']['val_label_dir']

    label_dict = Configs['dataset']['classes']
    train_dataset = customDataset(data_dir=train_data_dir,
                                  label_dir=train_label_dir,
                                  label_dict=label_dict)
    val_dataset = customDataset(data_dir=val_data_dir,
                                label_dir=val_label_dir,
                                label_dict=label_dict)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=Configs['train']['batch_size'],
                                  shuffle=True, num_workers=16, pin_memory=True)

    eval_loader = DataLoader(dataset=val_dataset, num_workers=16, shuffle=True, pin_memory=True)

    classifier = model(input_size=Configs['input_size'],
                                        n_channels = Configs['n_channels'],
                                        model_hyp=Configs['model'],
                                        classes=len(Configs['dataset']['classes'])).to('cuda')
    optimizer = torch.optim.Adam(classifier.parameters(),betas=(0.98,0.98),lr=Configs['optimizer']['init_lr'])
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(Configs['tensorboard']['runs_dir']+'train_board')    # Initilize tensorflow
    
    if Configs['warmup']==1:
        ### Warmup training
        warmup_steps = Configs['train']['warmup_steps']
        warmup_step = 0
        min_loss = 0.6
        val_loss = 0
        correct = 0
        total = 0
        val_accuracy = 0
        model_params = {
            'model_state_dict': classifier.state_dict(),
            'min_loss': min_loss,
            'val_accuracy': val_accuracy
        }
        while warmup_step < warmup_steps:
            classifier.train()
            for batch_index, (data,target,_) in enumerate(train_loader, 0):
                if warmup_step < warmup_steps:
                    optimizer.zero_grad()
            #     for batch_index, data in enumerate(train_loader, 0):
                    data, target = data.to('cuda'), target.to('cuda')
                    y = classifier(data)
            #         logger.debug(f"y size:{y.shape}, tatget size{target.shape}")
                    warmup_loss = criterion(y, target)
                    
                    warmup_loss.backward()
                    optimizer.step()
            #         logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
                    logger.info(f"Warmup Step: {warmup_step}, Warmup Loss: {warmup_loss}")
                    writer.add_scalar('Warmup Loss', warmup_loss, global_step=warmup_step)
                    warmup_step += 1

                if warmup_loss < min_loss:
                    with torch.no_grad():
                        for batch_index, (data,target,_) in enumerate(eval_loader, 0):
                            data, target = data.to('cuda'), target.to('cuda')
                            outputs = classifier(data)
                            loss = criterion(outputs, target)
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            total += target.size(0)  # Total number of samples
                            correct += (predicted == target).sum().item()  # Count correct predictions

                            val_loss /= len(eval_loader)
                            val_accuracy = 100 * correct / total
                        logger.info(f'Warmup Step: {warmup_step}, Warmup Loss: {warmup_loss}, Val Loss: {val_loss}, Accuracy:{val_accuracy:.2f}%')
                    min_loss = warmup_loss
                    torch.save(model_params,
                               Configs['checkpoint']['checkpoint_dir']+'inte_transformer_params_best.pth')
                        

            if warmup_step%5==0:
                torch.save(model_params, 
                           Configs['checkpoint']['checkpoint_dir']+'inte_transformer_params_last.pth')
            logger.info(f'min_loss: {min_loss}, Accuracy:{val_accuracy:.2f}%')
    else:
        ## load pre-trained model and train
        step = 0
        epochs = Configs['train']['n_epochs']
        state_dict = torch.load(Configs['checkpoint']['checkpoint_dir']+'0406_inte_transformer_params_best.pth')
        classifier.load_state_dict(state_dict)
        for epoch in range(epochs):
            # Training loop
            poly_lr_scheduler(optimizer, init_lr=Configs['optimizer']['init_lr'], iter=epoch, max_iter=epochs)
            for batch_index, (data,target,_) in enumerate(train_loader, 0):
                optimizer.zero_grad()
        #     for batch_index, data in enumerate(train_loader, 0):
                data, target = data.to('cuda'), target.to('cuda')
                y = classifier(data)
        #         logger.debug(f"y size:{y.shape}, tatget size{target.shape}")
                train_loss = criterion(y, target)
                
                train_loss.backward()
                optimizer.step()
        #         logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
                logger.info(f"Step: {step}, training Loss: {train_loss}")
                writer.add_scalar('Training Loss', train_loss, global_step=step)
                step += 1

            if epoch%5==0:
                val_loss = 0
                correct = 0
                total = 0
                accuracy = 0
                with torch.no_grad():
                    for batch_index, (data,target,_) in enumerate(eval_loader, 0):
                        data, target = data.to('cuda'), target.to('cuda')
                        outputs = classifier(data)
                        loss = criterion(outputs, target)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += target.size(0)  # Total number of samples
                        correct += (predicted == target).sum().item()  # Count correct predictions

                val_loss /= len(eval_loader)
                accuracy = 100 * correct / total
                writer.add_scalar('Validation Loss', val_loss, global_step=step)
                writer.add_scalar('Validation Accuracy', accuracy, global_step=step)
                logger.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

            torch.save(classifier.state_dict(), 
                       Configs['checkpoint']['checkpoint_dir']+'inte_transformer_params_latest.pth')
            if train_loss < min_loss:
                torch.save(classifier.state_dict(),
                           Configs['checkpoint']['checkpoint_dir']+'inte_transformer_params_best.pth')
                min_loss = train_loss


if __name__ == "__main__":
    logger = logging.getLogger(__name__)  # Use the current module's name
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    logger.addHandler(handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args = parser.parse_args(args=['configs/eeg_torch.yml'])
    # args, opts = parser.parse_known_args()
    # f = 'configs/eeg_pt.yml'
    with open(args.config_file, 'r') as file:
        configs = yaml.safe_load(file)

    # configs['optimizer']['init_lr']

    train(Configs=configs)

