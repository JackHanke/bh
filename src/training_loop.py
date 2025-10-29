# system imports
import os
import sys
import subprocess
import logging
import time
import pickle
import yaml

# training imports
import numpy as np
from tqdm import tqdm
import torch
from torchinfo import summary

# distributed training
import torch.distributed as dist  # NEW: Import for distributed training
import torch.multiprocessing as mp  # NEW: Import for multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP  # NEW: Import DDP wrapper
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 
from vae import VAE
from batching import custom_batcher, construct_batch
from dataset import HDF5Dataset

# 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 
def cleanup():
    dist.destroy_process_group()

## main training function for multi GPU training
def main_worker(
        rank: int, 
        world_size: int, 
        train_idxs: list[int],
        valid_idxs: list[int],
        data_path: str,
        model_path: str = None
    ):

    # setup environment
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # if main GPU, init logging
    if rank == 0:
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename='training.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    # load configs
    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    
    # read in config variables
    num_dumps = config['num_dumps']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    ## setup model
    model = VAE().to(device)

    # bring in model weights if model_path is provided
    if model_path is not None:
        model_dict_path = model_path
        model_dict = torch.load(model_dict_path)
        model.load_state_dict(model_dict)
        if rank == 0:
            model_weights_info_str = f"Loaded weights from: {model_path}"
            logger.info(model_weights_info_str)
            print(model_weights_info_str)
    else:
        if rank == 0:
            model_weights_info_str = f"Randomly initializing weights."
            logger.info(model_weights_info_str)
            print(model_weights_info_str)

    # get best validation from model, initially float('inf') for new model
    best_val_loss = model.best_val_seen
    
    if rank == 0:
        # summarize model 
        summary_str = summary(model, input_size=(batch_size, 8, 224, 48, 96))
        # model summary
        model_summary_str = '\n'+str(summary_str)
        logger.info(model_summary_str)
        print(model_summary_str)

        # training parameters
        training_hyperparams_str = f'''
        Training on dumps {start_dump} - {end_dump} 
            number of epochs: {num_epochs}
            batch size: {batch_size}
            logging device: {device}
        
        '''
        logger.info(training_hyperparams_str)
        print(training_hyperparams_str)
    
    # distribute model to GPU devices
    model = DDP(model, device_ids=[rank])
    
    # loss and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # distributed sampler to shared data across GPUs 
    # train_sampler = DistributedSampler(train_idxs, num_replicas=world_size, rank=rank, shuffle=True)
    # valid_sampler = DistributedSampler(valid_idxs, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataset = HDF5Dataset(data_path, dataset_type='train', percentage=0.9)
    valid_dataset = HDF5Dataset(data_path, dataset_type='valid', percentage=0.9)

    num_workers = os.cpu_count()  # Use all available CPU cores

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Automatically handles random sampling
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,  # Speeds up CPU-to-GPU memory copies
        num_replicas=world_size,
        rank=rank
    )

    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Automatically handles random sampling
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,  # Speeds up CPU-to-GPU memory copies
        num_replicas=world_size,
        rank=rank
    )

    # loss tracking
    train_losses, valid_losses = [], []

    ## training
    for epoch in range(num_epochs):
        train_start_time = time.time()
        model.train()

        # train loss tracking
        epoch_train_loss = []
        

        # batch training
        # train_batches = torch.utils.data.DataLoader(train_idxs, batch_size=batch_size, sampler=train_sampler)
        # prog_bar = tqdm(train_batches, disable=rank != 0)
        # for batch_indexes in prog_bar:

        prog_bar = tqdm(enumerate(train_loader), total=train_dataset.size, disable=rank != 0)
        for train_batch_num, (batch_data, label_data) in prog_bar:
            start = time.time()

            # send data to device
            batch_data, label_data = batch_data.to(device), label_data.to(device)
            
            # zero gradients
            optimizer.zero_grad()
            # compute prediction
            pred = model(batch_data)
            # compute loss
            loss = loss_fn(pred, label_data)
            # backprop and update gradients
            loss.backward()
            optimizer.step()
            # add loss to tracking
            loss_val = loss.item()
            epoch_train_loss.append(loss.item())

            # memory save maybe idk
            # batch_data = None
            # label_data = None
            # torch.cuda.empty_cache()

            # training batch logging
            if rank == 0: 
                batch_str = f'Train loss for epoch {epoch+1}, batch {train_batch_num}: {loss_val:.4f} in {time.time()-start:.2f}s'
                prog_bar.set_description(batch_str)
                logger.info(batch_str)
                # print(batch_str)

        train_loss_avg = sum(epoch_train_loss)/len(epoch_train_loss)
        train_losses.append(train_loss_avg)
        
        if rank == 0:
            train_str = f"Completed train loss for epoch {epoch+1}: {train_loss_avg:.4f} in {time.time()-train_start_time:.2f} s"
            prog_bar.set_description(train_str)
            logger.info(train_str)
            print(train_str)

        ## validation
        valid_start_time = time.time()
        model.eval()
        # loss tracking
        epoch_valid_loss = []
        

        ## batch validation
        # valid_batches = torch.utils.data.DataLoader(valid_idxs, batch_size=batch_size, sampler=valid_sampler)
        # prog_bar = tqdm(valid_batches, disable=rank != 0)
        # for batch_indexes in prog_bar:
        prog_bar = tqdm(enumerate(valid_loader), total=valid_dataset.size, disable=rank != 0)
        for valid_batch_num, (batch_data, label_data) in prog_bar:
            # send data to device
            batch_data, label_data = batch_data.to(device), label_data.to(device)
            
            # compute prediction
            with torch.no_grad():
                pred = model(batch_data)
            # compute loss
            loss = loss_fn(pred, label_data)
            # log validation loss
            epoch_valid_loss.append(loss.item())
            # validation batch logging
            if rank == 0: 
                batch_str = f'Validation loss for epoch {epoch+1}, batch {valid_batch_num}: {loss.item():.4f} in {time.time()-start:.2f}s'
                prog_bar.set_description(batch_str)
                logger.info(batch_str)
                # print(batch_str)

        if rank == 0:
            val_loss_avg = sum(epoch_valid_loss)/len(epoch_valid_loss)

            valid_str = f"Completed validation loss for epoch {epoch+1}: {val_loss_avg:.4f} in {time.time()-valid_start_time:.2f} s"
            prog_bar.set_description(train_str)
            logger.info(valid_str)
            print(valid_str)

            valid_losses.append(val_loss_avg)

            # save best model on rank 0
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                model.best_val_seen = best_val_loss # have model track best val for tracking
                model_save_path = os.environ['HOME'] + '/bh/harm2d/' + model.module.save_path
                model_save_info = f'Model saved at: {model_save_path}'
                model.module.save(model_save_path)
                logger.info(model_save_info)
                print(model_save_info)

    # Save training stats
    if rank == 0:
        with open(os.environ['HOME']+'/bh/harm2d/train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f)
        with open(os.environ['HOME']+'/bh/harm2d/valid_losses.pkl', 'wb') as f:
            pickle.dump(valid_losses, f)

    cleanup()