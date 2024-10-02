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




def data_clip(data_path:Path, result_path:Path, data_len:int, down_sample:int): # data_path = Path('./data/edf/train/'); 
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
        sub_label = str(file_path.parts[-3])
        file_name = str(file_path.name).split('.')[0]
    #     print(sub_label, file_path, file_name)
        raw = mne.io.read_raw_edf(file_path)
        raw.resample(down_sample)    # resampling to xHz
        sfreq = raw.info['sfreq']   # 100
    #     logger.info(freq)
        raw.crop(tmin=60)    # start from 60 secs
#         n_segments = int(np.floor(raw.times[-1] *sfreq/data_len))
        start, end = 0, data_len   # initilize slide window
        count = 0  # initilize num.of segments
        
        pd_frame = raw.to_data_frame(picks=['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF','EEG F4-REF', 
                                            'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
                                            'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                                            'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
                                            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
    #     channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF','EEG F4-REF', 
    #                                         'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 
    #                                         'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
    #                                         'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
    #                                         'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']    
#         print(pd_frame.shape)
    #     logger.debug(pe)  # Log as info
        while end <= pd_frame.shape[0]:
#         for i in range(n_segments):
#             start_time = i * data_len  # Start time of the segment in seconds
#             end_time = start_time + data_len  # End time of the segment in seconds

            # Extract the segment
            segment = pd_frame.iloc[start:end, 1:]
    #         logger.info(segment.shape, file_name)
            # normalization
            scaler = Normalizer()
            segment = scaler.fit_transform(segment.T).T
#             segment = segment.T + pe
            np.savetxt(f'{str(result_dir_path)}/{file_name}_{count+1}.csv', segment, header=','.join(channels), 
                       delimiter=',')
#             segment.to_csv(f'{str(result_dir_path)}/{file_name}_{count+1}.csv', index=False)

            label.loc[len(label)] = [f'{file_name}_{count+1}.csv', sub_label]
            start += data_len
            end += data_len
            count += 1
        
        raw.close()
    label.to_csv(f'../data/{stage}_label_{segment_length}.csv', index=False)
    
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
    args = parser.parse_args(args=['configs/abnormal_12000.yml'])
    # args, opts = parser.parse_known_args()
    # f = 'configs/eeg_pt.yml'
    with open(args.config_file, 'r') as file:
        configs = yaml.safe_load(file)
        
    # training dataset
    edf_data_path = Path(configs['dataset']['train_edf_dir'])  # need to modify
    result_dir_path = Path(configs['dataset']['train_data_dir'])
    segment_length = configs['input_size']
    down_sampling = configs['processing']['frequency']
    data_clip(edf_data_path, result_dir_path, segment_length, down_sampling)

    # eval dataset
    edf_data_path = Path(configs['dataset']['val_edf_dir'])  # need to modify
    result_dir_path = Path(configs['dataset']['val_data_dir'])
    segment_length = configs['input_size']
    data_clip(edf_data_path, result_dir_path, segment_length, down_sampling)