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
#
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
#
from datasets import HDF5Dataset
from models.vaes import VAE
from standardize_inputs import standardize, destandardize

def save_checkpoint(model, optimizer, scaler, epoch, loss, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'latent_dim': model.latent_dim,
        'input_channels': model.in_channels,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"checkpoint saved at {checkpoint_path}")
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"best model saved at {best_path}")
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

def compute_vae_loss(label: torch.Tensor, prediction: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    # loss configs
    W_RECON = 1.0
    BETA = 1e-5
    CLIP_VAL = 8.0 # limit for exp() and logvar to prevent inf

    # 
    L_recon = F.mse_loss(prediction, label, reduction='mean')
    
    logvar_c = torch.clamp(logvar, max=CLIP_VAL)
    L_KL = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
    
    L_total = (W_RECON * L_recon) + (BETA * L_KL)
    return L_total

def main():
    torch.autograd.set_detect_anomaly(True) 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open('../config.yaml', 'r') as file: config = yaml.safe_load(file)
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    start_dump = config['start_dump']
    end_dump = config['end_dump']

    avg_save_path = f'channel_wide_average.pt'
    avg_array = torch.load(avg_save_path)
    variance_save_path = f'channel_wide_variance.pt'
    variance_array = torch.load(variance_save_path)

    EPOCHS = 25
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
    LATENT_DIM = 256

    CHECKPOINT_DIR = os.path.join("checkpoints")
    SAVE_EVERY_N_EPOCHS = 3
    SAVE_BEST = True

    num_workers = 0

    data_path = os.getenv('SCRATCH')+"/data.hdf5"
    train_dataset = HDF5Dataset(data_path, dataset_type='train', percentage=0.9)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    print(f"Using device: {DEVICE}")
    input_shape = (8,224,48,96)
    
    model = VAE(
        input_shape=input_shape, 
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    # model summary
    summary_str = summary(model, input_size=(batch_size, 8, 224, 48, 96))
    model_summary_str = '\n'+str(summary_str)
    logger.info(model_summary_str)
    print(model_summary_str)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')

    # load in standardization arrays
    avg_array = torch.load(avg_save_path)
    variance_array = torch.load(variance_save_path)

    avg_total_loss = 0

    # training loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad() 
        model.train()
        
        prog_bar = tqdm(enumerate(train_loader), total=ceil(train_dataset.size/batch_size), )
        for train_batch_num, (batch_data, label_data) in prog_bar:
            #             
            standardized_batch = standardize(batch_data, avg_array, variance_array).to(DEVICE)
            # 
            prediction, mu, logvar = model(standardized_batch)
            #
            loss = compute_vae_loss(standardized_batch, prediction, mu, logvar)
            loss.backward()
            optimizer.step()
            #
            prog_bar.set_description(f'Batch loss: {loss.item()}')

        is_best = avg_total_loss < best_loss
        if is_best:
            best_loss = avg_total_loss
        
        if (epoch % SAVE_EVERY_N_EPOCHS) == 0 or is_best or epoch == EPOCHS-1:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                loss=None,
                checkpoint_dir=CHECKPOINT_DIR,
                is_best=is_best
            )

    print(f"Best loss achieved: {best_loss:.4e}")

if __name__ == "__main__":
    main()
