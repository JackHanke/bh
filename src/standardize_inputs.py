import os
import yaml
import numpy as np
from time import time
from tqdm import tqdm
import torch

from batching import custom_batcher, construct_batch

if __name__ == '__main__':
    # get configs
    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    num_dumps = config['num_dumps']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    # get indexes for training data
    train_idxs, valid_idxs = custom_batcher(
        batch_size=batch_size,
        num_dumps=num_dumps,
        split = 0.8,
        seed=1,
        start=start_dump,
        end=end_dump,
    )

    data_path = os.getenv('SCRATCH')+"/data.hdf5"

    train_batches = torch.utils.data.DataLoader(train_idxs, batch_size=batch_size)
    prog_bar = tqdm(train_batches)
    train_batch_num = 1
    for batch_indexes in prog_bar:
        start = time.time()

        batch_data, label_data = construct_batch(
            batch_indexes=batch_indexes, 
            data_path=data_path,
        )

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

