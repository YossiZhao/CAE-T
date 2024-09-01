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


def positional_encoding(max_length:int, d_model:int, model_type='sinusoidal'):
    """
    Generates positional encodings for a given maximum sequence length and model dimensionality.

    Args:
        max_length (int): The maximum length of the sequence.
        d_model (int): The dimensionality of the model.
        model_type (str): The type of positional encoding to use. Defaults to 'sinusoidal'.

    Returns:
        numpy.ndarray: The positional encoding matrix of shape (max_length, d_model).
    """

    if model_type == 'sinusoidal':
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(12000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        if pe.size % 2 != 0:
            pe[:, 1::2] = np.cos(position[:-1] * div_term)
    else:
        raise ValueError("Unsupported model_type: {}".format(model_type))

    return pe


def data_clip(data_path:Path, result_path:Path, label_path: str, data_len:int): # data_path = Path('./data/edf/train/'); 
                                                                # result_path = Path('./data/origin_csv/train/')
    if os.path.exists(result_dir_path):
        shutil.rmtree(result_dir_path)
    os.mkdir(result_dir_path)
    
    
    channels = ['Fp1', 'Fp2', 'F3','F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                'T5', 'T6', 'Fz', 'Cz', 'Pz']
    label_path = data_path
    stage = str(data_path.parts[-1])
    label = pd.DataFrame(columns=['csv_file','label'])
#     pe = positional_encoding(data_len, len(channels))    # 1000 needs to adjust according my study
    
    
    for file_path in data_path.glob('**/*.edf'):
        sub_label = str(file_path.parts[3])
        file_name = str(file_path.name).split('.')[0]
    #     print(sub_label, file_path, file_name)
        raw = mne.io.read_raw_edf(file_path)
        raw.resample(100)    # resampling to xHz
        sfreq = raw.info['sfreq']   # 100
    #     logger.info(freq)
        raw.crop(tmin=60)    # start from 60 secs
        n_segments = int(np.floor(raw.times[-1] *sfreq/data_len))
        pd_frame = raw.to_data_frame(picks=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF','EEG F4-REF', 
                                            'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
                                            'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                                            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
                                            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']) 
    #     print(pd_frame.shape)
    #     logger.debug(pe)  # Log as info
        for i in range(n_segments):
            start_time = i * data_len  # Start time of the segment in seconds
            end_time = start_time + data_len  # End time of the segment in seconds

            # Extract the segment
            segment = pd_frame.iloc[start_time:end_time, 1:]
    #         logger.info(segment.shape, file_name)

#             scaler = Normalizer()
#             segment = scaler.fit_transform(segment.T)
#             segment = segment + pe
    #         if segment.shape[0]>1000

    #         data, times = raw[:, start_time*freq:end_time*freq]
        #     print(times)
    #         df = raw.to_data_frame()
            segment.to_csv(f'{str(result_dir_path)}/{file_name}_{i+1}.csv', index=False)
    #         np.savetxt(f'./data/origin_csv/train/{file_name}_{i+1}.csv', segment, header=','.join(channels), 
    #                    delimiter=',')
            label.loc[len(label)] = [f'{file_name}_{i+1}.csv', sub_label]
        
        raw.close()
    label.to_csv(label_path, index=False)
    
if __name__ == "__main__":
    logger = logging.getLogger(__name__)  # Use the current module's name
    logging.basicConfig(level=logging.INFO)
#     logger.setLevel(logging.DEBUG)
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
        
    # training dataset
    edf_data_path = Path(configs['dataset']['train_edf_dir'])  # need to modify
    result_dir_path = Path(configs['dataset']['train_data_dir'])
    segment_length = configs['input_size']
    data_clip(edf_data_path, result_dir_path, configs['dataset']['train_label_dir'], segment_length)

    # eval dataset
    edf_data_path = Path(configs['dataset']['val_edf_dir'])  # need to modify
    result_dir_path = Path(configs['dataset']['val_data_dir'])
    segment_length = configs['input_size']
    data_clip(edf_data_path, result_dir_path, configs['dataset']['val_data_dir'], segment_length)