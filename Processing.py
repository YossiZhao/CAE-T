#!/usr/bin/env python
# coding: utf-8

import torch
from torch import Tensor, nn
from torch.types import Device, _size
from collections import OrderedDict

import os
import shutil
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import logging
import argparse
import yaml

from sklearn.preprocessing import Normalizer

from configs.config import configs




def data_clip(data_path:Path, result_dir_path:Path, data_len:int, down_sample:int):
    if os.path.exists(result_dir_path):
        shutil.rmtree(result_dir_path)
    os.mkdir(result_dir_path)
    
    
    channels = ['Fp1', 'Fp2', 'F3','F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                'T5', 'T6', 'Fz', 'Cz', 'Pz']
    label_path = data_path
    stage = str(data_path.parts[-1])
    label = pd.DataFrame(columns=['csv_file','label'])
    
    
    for file_path in data_path.glob('**/*.edf'):
        sub_label = str(file_path.parts[-3])
        file_name = str(file_path.name).split('.')[0]
        raw = mne.io.read_raw_edf(file_path)
        raw.resample(down_sample)    # resampling to xHz
        sfreq = raw.info['sfreq']   # 100
        raw.crop(tmin=60)    # start from 60 secs
        start, end = 0, data_len   # initilize slide window
        count = 0  # initilize num.of segments
        
        pd_frame = raw.to_data_frame(picks=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF','EEG F4-REF', 
                                            'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
                                            'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                                            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
                                            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
        while end <= pd_frame.shape[0]:


            # Extract the segment
            segment = pd_frame.iloc[start:end, 1:]

            segment.to_csv(f'{str(result_dir_path)}/{file_name}_{count+1}.csv', index=False)

            label.loc[len(label)] = [f'{file_name}_{count+1}.csv', sub_label]
            start += data_len
            end += data_len
            count += 1
        
        raw.close()
    label.to_csv(f'../data/{stage}_label_{segment_length}.csv', index=False)
    
if __name__ == "__main__":
    logger = logging.getLogger(__name__)  # Use the current module's name
    logging.basicConfig(level=logging.INFO)
    handler = logging.StreamHandler()

    logger.addHandler(handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="FILE", help="config file")

    args = parser.parse_args(args=['configs/encoderS+transformer.yml'])

    with open(args.config_file, 'r') as file:
        configs = yaml.safe_load(file)
    
    segment_length = configs['input_size']
    down_sampling = configs['processing']['frequency']
    # training dataset
    train_edf_data_path = Path(configs['dataset']['train_edf_dir'])  
    train_result_dir_path = Path(configs['dataset']['train_data_dir'])
    data_clip(train_edf_data_path, train_result_dir_path, segment_length, down_sampling)

    # val dataset
    val_edf_data_path = Path(configs['dataset']['val_edf_dir'])  
    val_result_dir_path = Path(configs['dataset']['val_data_dir'])
    data_clip(val_edf_data_path, val_result_dir_path, segment_length, down_sampling)
    
    # eval dataset
    eval_edf_data_path = Path(configs['dataset']['eval_edf_dir']) 
    eval_result_dir_path = Path(configs['dataset']['eval_data_dir'])
    data_clip(eval_edf_data_path, eval_result_dir_path, segment_length, down_sampling)