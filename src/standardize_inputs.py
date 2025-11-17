import os
import yaml
from math import ceil
import numpy as np
import time
from tqdm import tqdm
import torch

from batching import custom_batcher, construct_batch
from dataset import HDF5Dataset

if __name__ == '__main__':
    # get configs
    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    data_path = os.getenv('SCRATCH')+"/data.hdf5"
    avg_save_path = f'channel_wide_average.pt'
    variance_save_path = f'channel_wide_variance.pt'
    train_dataset = HDF5Dataset(data_path, dataset_type='train', percentage=0.9)

    # num_workers is number of CPUs used
    num_workers = 16

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    ## Calculate channel-wise mean
    if not os.path.isfile(avg_save_path):
        prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
        for train_batch_num, (batch_data, label_data) in prog_bar:
            start = time.time()
    
            # sum over channel
            sum_array = torch.sum(batch_data, dim=(0,2,3,4))
            
            # increment average by batch
            if train_batch_num == 0:
                temp_total = batch_data.shape[0]
                avg_array = sum_array/temp_total
            else:
                new_total = temp_total + batch_data.shape[0]
                avg_array = avg_array*(temp_total/new_total) + sum_array/new_total
                temp_total += batch_data.shape[0]
    

            batch_str = f'Batch {train_batch_num} completed in {time.time()-start:.4f}s.'
            prog_bar.set_description(batch_str)
            
        torch.save(f=avg_save_path, obj=avg_array)
    else:
        avg_array = torch.load(avg_save_path)
    
    print(f'avg_array = {avg_array}')
    
    ## Calculate channel-wise variance
    if not os.path.isfile(variance_save_path):
        prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
        for train_batch_num, (batch_data, label_data) in prog_bar:
            start = time.time()
            # 
            sum_array = torch.sum(
                torch.square(
                    batch_data - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.stack([avg_array for _ in range(batch_data.shape[0])],0),2),3),4)
                ), dim=(0,2,3,4))

            # increment average by batch
            if train_batch_num == 0:
                temp_total = batch_data.shape[0]
                variance_array = sum_array/temp_total
            else:
                new_total = temp_total + batch_data.shape[0]
                variance_array = avg_array*(temp_total/new_total) + sum_array/new_total
                temp_total += batch_data.shape[0]
    
            batch_str = f'Batch {train_batch_num} completed in {time.time()-start:.4f}s.'
            prog_bar.set_description(batch_str)

        torch.save(f=variance_save_path, obj=variance_array)
    else:
        variance_array = torch.load(variance_save_path)
        
    print(f'variance_array = {variance_array}')
    



