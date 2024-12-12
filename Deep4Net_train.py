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
import time

from sklearn.metrics import confusion_matrix

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

from configs.config import configs


from braindecode.models import Deep4Net


# Transform signal
def transform(data:Tensor, mean:Tensor, std:Tensor):
    normalized_data = (data - mean) / std
    return normalized_data

# ### Dataset

class customDataset(Dataset):
    def __init__(self, data_dir:str, label_dir:str, label_dict:dict, mean: list, std: list, transform=None):
        
        self.data_dir = data_dir   # './data/seg_csv/train'
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
        
#         label = torch.tensor(int(self.label_dict[self.annotations.iloc[index,1]]))
        label = self.annotations.loc[self.annotations['csv_file'] == file_name, ['label']].to_string(index=False,header=False)
        label = torch.tensor(int(self.label_dict[label]))
        
        
        if self.transform:
            data = self.transform(data, self.mean, self.std)
            
        return (data, label, file_name)


class model(nn.Module):
    def __init__(self, input_size: int, n_channels: int, model_hyp: dict, classes: int):
        super(model, self).__init__()
        self.de = Deep4Net(in_chans=n_channels, n_classes=classes,
                 input_window_samples=input_size,
                 n_filters_time=25, n_filters_spat=25,
                 stride_before_pool=True,
                 n_filters_2=n_channels * 2,
                 n_filters_3=n_channels * 4,
                 n_filters_4=n_channels * 8,
                 final_conv_length=1,
                 add_log_softmax=False, )
        self.linear = nn.Linear(2*437, classes)
        
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
        
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
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
        z = self.de(z)
        z = torch.flatten(z, 1)
        y = self.linear(z)
        return y

    

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
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    curr_lr_rt = optimizer.param_groups[0]['lr']
    logger.info(f'lr=: {curr_lr_rt}')
    return curr_lr_rt


def evaluate_model(model, dataloader, criterion):
    signal_to_case_map = []  # Map each signal index to a case ID, e.g., [case1, case1, case2, ...]
    for data, label, file_name in dataloader:
        signal_to_case_map.append(file_name[0].split('_')[0])
    
    # Predict a single signal
    def predict_signal(model, signal):
        with torch.no_grad():
            output = model(signal)
            # For NNGNet
    #         output = model(signal.unsqueeze(1)).to('cpu')
        
        prediction = torch.argmax(output, dim=1).item()  # Get the predicted class
        return prediction

    # Calculate sensitivity, specificity, and accuracy
    def calculate_metrics(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.tolist()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.tolist()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return sensitivity, specificity, accuracy

    # Evaluate per-signal metrics
    def evaluate_per_signal(model, dataset):
        y_true, y_pred = [], []
        for signal, label, _ in dataset:
            signal= signal.to('cuda')
            if isinstance(label, torch.Tensor):
                label = label.item()
            prediction = predict_signal(model, signal)
            y_true.append(label)
            y_pred.append(prediction)
        sensitivity, specificity, accuracy = calculate_metrics(y_true, y_pred)
        return sensitivity, specificity, accuracy, y_true, y_pred

    # Aggregate signals for per-case metrics
    def evaluate_per_case(y_true, y_pred, signal_to_case_map):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.tolist()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.tolist()

        case_results = {}
        for signal_idx, case_id in enumerate(signal_to_case_map):
            if case_id not in case_results:
                case_results[case_id] = {'true': [], 'pred': []}
            case_results[case_id]['true'].append(y_true[signal_idx])
            case_results[case_id]['pred'].append(y_pred[signal_idx])

        # Per-case metrics
        y_true_case, y_pred_case = [], []
        for case_id, results in case_results.items():
            # Majority vote for case prediction
            true_label = max(set(results['true']), key=results['true'].count)
            pred_label = max(set(results['pred']), key=results['pred'].count)
            y_true_case.append(true_label)
            y_pred_case.append(pred_label)

        sensitivity, specificity, accuracy = calculate_metrics(y_true_case, y_pred_case)
        return sensitivity, specificity, accuracy, y_true_case, y_pred_case

    # Per-signal evaluation
    logger.info("Evaluating per-signal metrics...")
    sensitivity_signal, specificity_signal, accuracy_signal, y_true_signal, y_pred_signal = evaluate_per_signal(
        model, dataloader)
    logger.info(f"Per-Signal Sensitivity: {sensitivity_signal:.4f}, Specificity: {specificity_signal:.4f}, Accuracy {accuracy_signal:.4f}")

    # Per-case evaluation
    logger.info("Evaluating per-case metrics...")
    sensitivity_case, specificity_case, accuracy_case, y_true_case, y_pred_case = evaluate_per_case(
        y_true_signal, y_pred_signal, signal_to_case_map)
    logger.info(f"Per-Case Sensitivity: {sensitivity_case:.4f}, Specificity: {specificity_case:.4f}, Accuracy: {accuracy_case:.4f}")
    return sensitivity_signal, specificity_signal, accuracy_signal, sensitivity_case, specificity_case, accuracy_case
    
    

def train(Configs:dict):
    train_data_dir = Configs['dataset']['train_data_dir']
    train_label_dir = Configs['dataset']['train_label_dir']

    val_data_dir = Configs['dataset']['val_data_dir']
    val_label_dir = Configs['dataset']['val_label_dir']

    label_dict = Configs['dataset']['classes']
    
    mean = Configs['dataset']['mean']
    std = Configs['dataset']['std']
    
    model_name = Configs['model']['name']
    
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
    

#     input_layer = nn.Sequential(
# #         nn.Embedding(num_embeddings=10000, embedding_dim=512),
# #         PositionalEncoding(d_model=512, dropout=0.1, max_len=5000)
#         PositionalEncoding(d_model=Configs['n_channels'], max_len=Configs['input_size'])
#     ).to('cuda')
    if Configs['checkpoint']['weights'] is not None:
        print(f'loading pre-trained model...')
        classifier = torch.load(Configs['checkpoint']['checkpoint_dir']+Configs['checkpoint']['weights'])
#         classifier.load_state_dict(state_dict)
    else:
        classifier = model(input_size=Configs['input_size'],
                                        n_channels = Configs['n_channels'],
                                        model_hyp=Configs['model'],
                                        classes=len(Configs['dataset']['classes'])).to('cuda')
    
    optimizer = torch.optim.Adam(classifier.parameters(),lr=Configs['optimizer']['init_lr'], weight_decay=Configs['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    writer = SummaryWriter(Configs['tensorboard']['runs_dir']+f'{datetime.now().strftime("%y%m%d%H%M")}_{model_name}_train_board')   # Initilize tensorboard
    
    min_loss = 0.5
    best_accuracy = 0.75
    
    start_time = time.time()
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
#                     input_pe = input_layer(data)
                    y = classifier(data)
            #         logger.debug(f"y size:{y.shape}, target size{target.shape}")
                    warmup_loss = criterion(y, target)
                    
                    warmup_loss.backward()
                    optimizer.step()
            #         logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
                    logger.info(f"Warmup Step: {warmup_step}, Warmup Loss: {warmup_loss}")
                    writer.add_scalar('Warmup Loss', warmup_loss, global_step=warmup_step)
#                     writer.flush()
                    warmup_step += 1

                if warmup_loss < min_loss:  # evaluate model
                    min_loss = warmup_loss
    
    
    #Start training
    ## load pre-trained model and train
    step = 0
    epochs = Configs['train']['n_epochs']
    
    for epoch in range(epochs):
        # Training loop
        curr_lr_rt = poly_lr_scheduler(optimizer, init_lr=Configs['optimizer']['init_lr'], iter=epoch, max_iter=epochs)
        for batch_index, (data,target,_) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            data, target = data.to('cuda'), target.to('cuda')
            y = classifier(data)
            train_loss = criterion(y, target)

            train_loss.backward()
            optimizer.step()
            logger.info(f"Epoch: {epoch+1}, Step: {step}, training Loss: {train_loss}")
            writer.add_scalar('Training Loss', train_loss, global_step=step)
#             writer.flush()
            step += 1

        sensitivity_signal, specificity_signal, accuracy_signal, sensitivity_case, specificity_case, accuracy_case = evaluate_model(classifier, eval_loader, criterion)
#         writer.add_scalar('Validation Loss', val_loss, global_step=step)
#         writer.add_scalar('Validation Accuracy', val_accuracy, global_step=step)
#         writer.add_scalar('Validation Accuracy with epoch', val_accuracy, global_step=epoch)
        
#         if val_accuracy > best_accuracy:
#             best_accuracy = val_accuracy
        torch.save(classifier, Configs['checkpoint']['checkpoint_dir']+ f'{datetime.now().strftime("%y%m%d%H%M")}_{model_name}_params_epoch_{epoch+1}_acc{accuracy_signal:04f}.pth')
        writer.add_hparams({'lr': curr_lr_rt, 'bsize': Configs['train']['batch_size'], 'input_size': Configs['input_size'], 'epoch': epoch+1},{'sensitivity_signal': sensitivity_signal*100, 'specificity_signal': specificity_signal*100, 'accuracy_signal': accuracy_signal*100, 'sensitivity_case': sensitivity_case*100, 'specificity_case': specificity_case*100, 'accuracy_case': accuracy_case*100})
#             writer.flush()
            
            
    end_time = time.time()        
    writer.close()
    
    # Convert elapsed time to hours, minutes, and seconds
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    # Format the time as "xh ym zs"
#     formatted_time = f"{hours}h {minutes}m {seconds}s"
    logger.info(f"Training time: {hours}h {minutes}m {seconds}s")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args = parser.parse_args(args=['configs/Deep4Conv.yml'])
    
    with open(args.config_file, 'r') as file:
        configs = yaml.safe_load(file)
    model_name = configs['model']['name']
    logger = logging.getLogger(__name__)  # Use the current module's name
    logging.basicConfig(filename=f'../logs/{datetime.now().strftime("%y%m%d%H%M")}_{model_name}.log', level=logging.INFO)
#     logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    logger.addHandler(handler)
    

    # configs['optimizer']['init_lr']

    train(Configs=configs)

