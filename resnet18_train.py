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

from models.pe import PositionalEncoding
from models.resnet1D import *
from models.transformer_encoder import transformer_classifier



# Transform signal
def transform(data:Tensor, mean:Tensor, std:Tensor):
    normalized_data = (data - mean) / std
    return normalized_data

# ### Dataset

class customDataset(Dataset):
    def __init__(self, data_dir:str, label_dir:str, label_dict:dict, mean: list, std: list, transform=None):
#         self.annotations = pd.read_csv(label_dir)
        self.data_dir = data_dir   # './data/origin_csv/train'
        self.label_dir = label_dir
        self.transform = transform
        self.files = os.listdir(self.data_dir)
        self.annotations = pd.read_csv(self.label_dir)
        self.label_dict = label_dict
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.files[index])
        data = pd.read_csv(data_path)
        data = torch.tensor(data.values, dtype=torch.float32)
        file_name = self.files[index]
        
        label = torch.tensor(int(self.label_dict[self.annotations.iloc[index,1]]))
        
        if self.transform:
            data = self.transform(data, self.mean, self.std)
            
        return (data, label, file_name)



### Define classifier
# class model(nn.Module):
#     def __init__(self, input_size: int, n_channels: int, model_hyp: dict, classes: int):
#         super(model, self).__init__()
#         self.ae = resnet18(n_channels=n_channels, groups=n_channels, num_classes=classes, d_model=model_hyp['d_model'])
# #         self.transformer_encoder = transformer_classifier(input_size, n_channels, model_hyp, classes)

#     def forward(self, x):
#         z = self.ae(x)
# #         z = self.transformer_encoder(z)
#         return z

class model(nn.Module):
    def __init__(self, input_size: int, n_channels: int, model_hyp: dict, classes: int):
        super(model, self).__init__()
        self.ae = resnet18(n_channels=n_channels, groups=n_channels, num_classes=classes, d_model=model_hyp['d_model'])
        self.transformer_encoder = transformer_classifier(input_size, n_channels, model_hyp, classes)
        
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
        
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print('Complete initiate parameters')

    def forward(self, x):
#         z = self.pe(x)
        z = x.transpose(-1,-2)
        z = self.ae(z)
#         z = self.transformer_encoder(z)
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


def evaluate_model(model, eval_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    val_loss = 0
    correct = 0
    total = 0
    val_accuracy = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_index, (data,target,_) in enumerate(eval_loader, 0):
#         for data, labels in val_loader:
            data, target = data.to('cuda'), target.to('cuda')
            input_pe = input_layer(data)
            outputs = model(input_pe)
            loss = criterion(outputs, target)
            val_loss += loss.item()
#             outputs = model(data)  # Forward pass to get logits
            probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            _, predicted = torch.max(probabilities, 1)  # Get the predicted class
        
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        val_loss /= len(eval_loader)
        val_accuracy = 100 * correct / total
    logger.info(f'Val Loss: {val_loss}, Accuracy:{val_accuracy:.2f}%')
    
    

def train(Configs:dict):
    train_data_dir = Configs['dataset']['train_data_dir']
    train_label_dir = Configs['dataset']['train_label_dir']

    val_data_dir = Configs['dataset']['val_data_dir']
    val_label_dir = Configs['dataset']['val_label_dir']

    label_dict = Configs['dataset']['classes']
    
    mean = Configs['dataset']['mean']
    std = Configs['dataset']['std']
    
    train_dataset = customDataset(data_dir=train_data_dir,
                                  label_dir=train_label_dir,
                                  label_dict=label_dict,
                                 mean=mean, std=std,
                                 transform=transform)
    val_dataset = customDataset(data_dir=val_data_dir,
                                label_dir=val_label_dir,
                                label_dict=label_dict,
                               mean=mean, std=std,
                               transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=Configs['train']['batch_size'],
                              shuffle=Configs['dataset']['shuffle'], 
                              num_workers=Configs['dataset']['num_workers'], pin_memory=True)

    eval_loader = DataLoader(dataset=val_dataset, num_workers=Configs['dataset']['num_workers'], 
                             shuffle=Configs['dataset']['shuffle'], pin_memory=True)
    

    input_layer = nn.Sequential(
#         nn.Embedding(num_embeddings=10000, embedding_dim=512),
#         PositionalEncoding(d_model=512, dropout=0.1, max_len=5000)
        PositionalEncoding(d_model=Configs['n_channels'], max_len=Configs['input_size'])
    ).to('cuda')

    classifier = model(input_size=Configs['input_size'],
                                        n_channels = Configs['n_channels'],
                                        model_hyp=Configs['model'],
                                        classes=len(Configs['dataset']['classes'])).to('cuda')
    
    optimizer = torch.optim.Adam(classifier.parameters(),lr=Configs['optimizer']['init_lr'], weight_decay=Configs['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(Configs['tensorboard']['runs_dir']+
                           f'{datetime.now().strftime("%y%m%d%H%M")}_train_board')    # Initilize tensorflow
    min_loss = 0.3
    
    if Configs['warmup']==1:
        ### Warmup training
        warmup_steps = Configs['train']['warmup_steps']
        warmup_step = 0

        while warmup_step < warmup_steps:
            
            for batch_index, (data,target,_) in enumerate(train_loader, 0):
                classifier.train()
                if warmup_step < warmup_steps:
                    optimizer.zero_grad()
            #     for batch_index, data in enumerate(train_loader, 0):
                    data, target = data.to('cuda'), target.to('cuda')
                    input_pe = input_layer(data)
                    y = classifier(input_pe)
            #         logger.debug(f"y size:{y.shape}, tatget size{target.shape}")
                    warmup_loss = criterion(y, target)
                    
                    warmup_loss.backward()
                    optimizer.step()
            #         logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
                    logger.info(f"Warmup Step: {warmup_step}, Warmup Loss: {warmup_loss}")
                    writer.add_scalar('Warmup Loss', warmup_loss, global_step=warmup_step)
                    warmup_step += 1

                if warmup_loss < min_loss:  # evaluate model
#                     logger.info(f'Evaluation: Warmup Loss: {warmup_loss}.')
                    evaluate_model(classifier, eval_loader, criterion)
                    min_loss = warmup_loss
#                     model_params = {
#                         'model_state_dict': classifier.state_dict(),
#                         'val_loss': val_loss,
#                         'val_accuracy': val_accuracy
#                     }
                    torch.save(classifier.state_dict(), Configs['checkpoint']['checkpoint_dir']+
                               f'{datetime.now().strftime("%y%m%d%H%M")}_resnet18_params_best.pth')
                        
        torch.save(classifier.state_dict(), Configs['checkpoint']['checkpoint_dir']+
                   f'{datetime.now().strftime("%y%m%d%H%M")}_resnet18_params_latest.pth')
        logger.info(f'min_loss: {min_loss}')
        evaluate_model(classifier, eval_loader, criterion)
    
    
    #Start training
    ## load pre-trained model and train
    step = 0
    epochs = Configs['train']['n_epochs']
    if Configs['checkpoint']['weights'] is not None:
        state_dict = torch.load(Configs['checkpoint']['checkpoint_dir']+Configs['checkpoint']['weights'])
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
            logger.info(f"Epoch: {epoch+1}, Step: {step}, training Loss: {train_loss}")
            writer.add_scalar('Training Loss', train_loss, global_step=step)
            step += 1

        if epoch%5==0:
#             logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
            evaluate_model(classifier, eval_loader, criterion)
#             writer.add_scalar('Validation Loss', val_loss, global_step=step)
#             writer.add_scalar('Validation Accuracy', accuracy, global_step=step)
            

        
        if train_loss < min_loss:
            evaluate_model(classifier, eval_loader, criterion)
            min_loss = train_loss
            torch.save(classifier.state_dict(), Configs['checkpoint']['checkpoint_dir']+
                       f'{datetime.now().strftime("%y%m%d%H%M")}_resnet18_params_best.pth')
            
    torch.save(classifier.state_dict(), Configs['checkpoint']['checkpoint_dir']+
               f'{datetime.now().strftime("%y%m%d%H%M")}_resnet18_params_latest.pth')
    evaluate_model(classifier, eval_loader, criterion)
            


if __name__ == "__main__":
    logger = logging.getLogger(__name__)  # Use the current module's name
    logging.basicConfig(filename=f'../logs/{datetime.now().strftime("%y%m%d%H%M")}_resnet18.log', level=logging.INFO)
#     logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    logger.addHandler(handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args = parser.parse_args(args=['configs/abnormal_12000.yml'])
    # args, opts = parser.parse_known_args()
    # f = 'configs/eeg_pt.yml'
    with open(args.config_file, 'r') as file:
        configs = yaml.safe_load(file)

    # configs['optimizer']['init_lr']

    train(Configs=configs)

