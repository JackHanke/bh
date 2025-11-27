#
import os
import numpy as np
from tqdm import tqdm
import yaml
from math import ceil
#
import logging
import pickle
import yaml
import time
from datetime import datetime
#
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
#
from datasets import HDF5Dataset
from models.vaes import VAE, VAE_loss
from standardize_inputs import standardize, destandardize

def save_checkpoint(model, checkpoint_dir, epoch, experiment_start_time):
    logger = logging.getLogger(__name__) 
    checkpoint_path = os.path.join(checkpoint_dir, f'vae_{experiment_start_time}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Checkpoint saved at: {checkpoint_path}")

def train_vae():
    torch.autograd.set_detect_anomaly(True)

    experiment_start_time = datetime.now()
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f'logs/experiment-{experiment_start_time}.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']
    # 
    avg_save_path = config['avg_save_path']
    variance_save_path = config['variance_save_path']
    avg_array = torch.load(avg_save_path)
    variance_array = torch.load(variance_save_path)

    LATENT_DIM = config['latent_dim']
    # 
    EPOCHS = config['num_epochs']
    LEARNING_RATE = config['learning_rate']
    BATCH_SIZE = config['batch_size']
    CHECKPOINT_DIR = 'models/checkpoints'
    SAVE_EVERY_N_EPOCHS = config['save_every_n_epochs']

    SAVE_BEST = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 10
    input_shape = (8,224,48,96)

    data_path = os.getenv('SCRATCH')+"/data.hdf5"
    train_dataset = HDF5Dataset(
        data_path, 
        dataset_type='train', 
        percentages=(0.8, 0.1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    validation_dataset = HDF5Dataset(
        data_path, 
        dataset_type='valid', 
        percentages = (0.8, 0.1)
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Starting experiment on: {experiment_start_time}")
    logger.info(f"Using device: {DEVICE}")
    
    model = VAE(
        input_shape=input_shape, 
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    # model summary
    summary_str = summary(model, input_size=(batch_size, 8, 224, 48, 96))
    model_summary_str = '\n'+str(summary_str)
    logger.info(model_summary_str)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')

    # load in standardization arrays
    avg_array = torch.load(avg_save_path)
    variance_array = torch.load(variance_save_path)


    # main loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad() 

        #
        training_losses, validation_losses = [], []
        training_reconstruction_losses, validation_reconstruction_losses = [], []
        training_kl_losses, validation_kl_losses = [], []

        ## train
        model.train()
        prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
        for train_batch_num, (batch_data, _) in prog_bar:
            #             
            standardized_batch = standardize(batch_data, avg_array, variance_array).to(DEVICE)
            # 
            prediction, mu, logvar = model(standardized_batch)
            #
            loss, reconstruction_loss, kl_loss = VAE_loss(standardized_batch, prediction, mu, logvar)
            loss.backward()
            # 
            training_losses.append(loss.item())
            training_reconstruction_losses.append(reconstruction_loss.item())
            training_kl_losses.append(kl_loss.item())
            
            optimizer.step()
            #
            prog_bar.set_description(f'Epoch {epoch}, Batch {train_batch_num} Train Loss: {loss.item():.5f}')

        logger.info(f'Epoch {epoch} Train Loss: {sum(training_losses)/len(training_losses)}\nTrain Reconstruction Loss: {sum(training_reconstruction_losses)/len(training_reconstruction_losses)}\nTrain KL Loss: {sum(training_kl_losses)/len(training_kl_losses)}')
        
        ## validation
        model.eval()
        prog_bar = tqdm(enumerate(validation_loader), total=ceil(validation_dataset.size/batch_size), )
        for validation_batch_num, (batch_data, _) in prog_bar:
            #             
            standardized_batch = standardize(batch_data, avg_array, variance_array).to(DEVICE)
            # 
            prediction, mu, logvar = model(standardized_batch)
            #
            loss, reconstruction_loss, kl_loss = VAE_loss(standardized_batch, prediction, mu, logvar)
            # 
            validation_losses.append(loss.item())
            validation_reconstruction_losses.append(reconstruction_loss.item())
            validation_kl_losses.append(kl_loss.item())
            # 
            prog_bar.set_description(f'Epoch {epoch}, Batch {train_batch_num} Valid Loss: {loss.item():.5f}')

        avg_total_loss = sum(validation_losses)/len(validation_losses)
        logger.info(f'Epoch {epoch} Validation Loss: {avg_total_loss}\nValidation Reconstruction Loss: {sum(validation_reconstruction_losses)/len(validation_reconstruction_losses)}\nValidation KL Loss: {sum(validation_kl_losses)/len(validation_kl_losses)}')

        # 
        is_best = avg_total_loss < best_loss
        if is_best:
            best_loss = avg_total_loss

        # checkpoint
        if (epoch % SAVE_EVERY_N_EPOCHS) == 0 or is_best or epoch == EPOCHS-1:
            save_checkpoint(
                model=model,
                checkpoint_dir=CHECKPOINT_DIR,
                epoch=epoch,
                experiment_start_time=experiment_start_time,
            )

    # print(f"Best loss achieved: {best_loss:.4e}")

if __name__ == "__main__":
    train_vae()
