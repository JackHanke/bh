import os
import yaml
import numpy as np
from time import time
from tqdm import tqdm
import torch

from batching import custom_batcher, construct_batch
from dataset import HDF5Dataset

if __name__ == '__main__':
    # get configs
    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    num_dumps = config['num_dumps']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    data_path = os.getenv('SCRATCH')+"/data.hdf5"
    train_dataset = HDF5Dataset(data_path, dataset_type='train', percentage=0.9)

    num_workers = os.cpu_count()  # Use all available CPU cores

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Automatically handles random sampling
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True  # Speeds up CPU-to-GPU memory copies
    )

    prog_bar = tqdm(enumerate(train_loader), total=train_dataset.size)
    for train_batch_num, (batch_data, label_data) in prog_bar:
        start = time.time()

        # sum over channel
        sum_array = np.sum(batch_data, axis=(0,2,3,4))
        
        # increment average by batch
        if train_batch_num == 1:
            temp_total = batch_data.shape[0]
            avg_array = sum_array/temp_total
        else:
            new_total = temp_total + batch_data.shape[0]
            avg_array = avg_array*(temp_total/new_total) + sum_array/new_total
            temp_total += batch_data.shape[0]

        batch_str = f'Batch {train_batch_num} completed in {time.time()-start:.2f}s, averages per channel: {avg_array}'

    np.save(file=f'channel_wide_average.npy', arr=avg_array)

