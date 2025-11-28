import os
import yaml
from math import ceil
import numpy as np
import time
from tqdm import tqdm
import torch

# standardize data for given avg_array and variance_array
def standardize(data: torch.Tensor, avg_array: torch.Tensor, variance_array: torch.Tensor, device='cpu'):
    transformed_avg_array = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.stack([avg_array for _ in range(data.shape[0])],0),2),3),4).to(device)
    transformed_variance_array = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.stack([variance_array for _ in range(data.shape[0])],0),2),3),4).to(device)
    
    standardized_data = (data - transformed_avg_array)/torch.sqrt(transformed_variance_array)
    return standardized_data

# de-standardize data 
def destandardize(data: torch.Tensor, avg_array: torch.Tensor, variance_array: torch.Tensor, device = 'cpu'):
    transformed_avg_array = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.stack([avg_array for _ in range(data.shape[0])],0),2),3),4).to(device)
    transformed_variance_array = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.stack([variance_array for _ in range(data.shape[0])],0),2),3),4).to(device)

    destandardized_data = (data * torch.sqrt(transformed_variance_array)) + transformed_avg_array
    return destandardized_data

if __name__ == '__main__':
    from datasets import HDF5Dataset
    
    # get configs
    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    data_path = os.getenv('SCRATCH')+"/data.hdf5"
    train_dataset = HDF5Dataset(data_path, dataset_type='train', percentages=(0.8,0.1))
    avg_save_path = f'channel_wide_average.pt'
    variance_save_path = f'channel_wide_variance.pt'

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
        num_datapoints = 0
        prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
        for train_batch_num, (batch_data, label_data) in prog_bar:
            start = time.time()
    
            # sum over everything but channel
            sum_array = torch.sum(batch_data, dim=(0,2,3,4))
            
            # increment average by batch
            num_datapoints += batch_data.shape[0]
            if train_batch_num == 0:
                avg_array = sum_array
            else:
                avg_array += sum_array

            batch_str = f'Batch {train_batch_num} completed in {time.time()-start:.4f}s.'
            prog_bar.set_description(batch_str)
        avg_array = avg_array / (num_datapoints*config['data_x']*config['data_y']*config['data_z'])
        torch.save(f=avg_save_path, obj=avg_array)
    else:
        avg_array = torch.load(avg_save_path)
    
    print(f'avg_array = {avg_array}')
    
    ## Calculate channel-wise variance
    if not os.path.isfile(variance_save_path):
        num_datapoints = 0
        prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
        for train_batch_num, (batch_data, label_data) in prog_bar:
            start = time.time()
            # 
            sum_array = torch.sum(
                torch.square(
                    batch_data - torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.stack([avg_array for _ in range(batch_data.shape[0])],0),2),3),4)
                ), dim=(0,2,3,4))

            # increment average by batch
            num_datapoints += batch_data.shape[0]
            if train_batch_num == 0:
                variance_array = sum_array
            else:
                variance_array += sum_array
    
            batch_str = f'Batch {train_batch_num} completed in {time.time()-start:.4f}s.'
            prog_bar.set_description(batch_str)
        variance_array = variance_array / (num_datapoints*config['data_x']*config['data_y']*config['data_z'])
        torch.save(f=avg_save_path, obj=avg_array)
        torch.save(f=variance_save_path, obj=variance_array)
    else:
        variance_array = torch.load(variance_save_path)
        
    print(f'variance_array = {variance_array}')

    prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
    for train_batch_num, (batch_data, label_data) in prog_bar:
        standardized_batch = standardize(batch_data, avg_array, variance_array)
        print(f'standardized mean: {torch.mean(standardized_batch)} stdev: {torch.sqrt(torch.var(standardized_batch))}')
        for i in range(8):
            print(f'mean channel {i}: {torch.mean(standardized_batch[:, i])} stdev: {torch.sqrt(torch.var(standardized_batch[:, i]))}')
        break
    



