import os
import glob
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



DATA_DIR = "/storage/scratch1/4/skumar680/5000run/data_output/" 
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8 
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
LATENT_DIM = 256 

# loss fn weights
W_RECON = 1.0   
BETA = 1e-5       

# normalization
EPSILON = 1e-12    
CLIP_VAL = 8.0       # limit for exp() and logvar to prevent inf

# the model randomly stops so added checkpoints
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
SAVE_EVERY_N_EPOCHS = 3  
SAVE_BEST = True 

RHO_CHANNEL = 0
UGAS_CHANNEL = 1
UU_CHANNELS = [2, 3, 4, 5]
B_CHANNELS = [6, 7, 8, 9]
LOG_CHANNELS = [RHO_CHANNEL, UGAS_CHANNEL]
STD_CHANNELS = UU_CHANNELS + B_CHANNELS


def get_normalization_stats(loader, device):
    """Calculates mean/std/min"""
    print("Calculating normalization statistics...")
    STATS_FILE = os.path.join(DATA_DIR, "norm_stats.pth")
    
    if os.path.exists(STATS_FILE):
        print(f"Loading normalization stats from {STATS_FILE}...")
        stats = torch.load(STATS_FILE, map_location=device, weights_only=True)
    else:
        print("Stats file not found. Scanning dataset (this may take a while)...")
        
        # Get shape from first item
        first_prims = next(iter(loader))
        _, _, D, H, W = first_prims.shape
        num_pixels_per_dump = D * H * W
        
        sum_ = torch.zeros(len(STD_CHANNELS), device=device)
        sum_sq_ = torch.zeros(len(STD_CHANNELS), device=device)
        n_pixels = 0
        log_mins = [float('inf')] * len(LOG_CHANNELS)
        
        with torch.no_grad():
            for prims_batch in tqdm(loader, desc="Scanning Dataset"):
                prims = prims_batch.to(device)                
                std_data = prims[:, STD_CHANNELS, ...]
                sum_ += std_data.sum(dim=(0, 2, 3, 4))
                sum_sq_ += (std_data**2).sum(dim=(0, 2, 3, 4))
                n_pixels += std_data.shape[0] * num_pixels_per_dump
                
                for i, channel_idx in enumerate(LOG_CHANNELS):
                    log_mins[i] = min(log_mins[i], prims[:, channel_idx, ...].min().item())
                    
        mean = sum_ / n_pixels
        variance = (sum_sq_ / n_pixels) - mean**2
        std = torch.sqrt(F.relu(variance) + 1e-10) 
        
        stats = {
            'mean': mean,
            'std': std,
            'log_mins': torch.tensor(log_mins, device=device)
        }
        torch.save(stats, STATS_FILE)

    print("Normalization stats loaded.")
    print(f"  Mean (uu, B): {stats['mean'].cpu().numpy()}")
    print(f"  Std (uu, B): {stats['std'].cpu().numpy()}")
    print(f"  Min (rho, u_gas): {stats['log_mins'].cpu().numpy()}")
    return stats

def normalize_prims(prims, stats, device):
    prims = prims.to(device)
    prims_norm = torch.clone(prims)
    
    mean = stats['mean'].view(-1, 1, 1, 1)
    std = stats['std'].view(-1, 1, 1, 1)
    prims_norm[:, STD_CHANNELS, ...] = (prims[:, STD_CHANNELS, ...] - mean) / (std + EPSILON)
    
    for i, channel_idx in enumerate(LOG_CHANNELS):
        min_val = stats['log_mins'][i]
        shifted_data = prims[:, channel_idx, ...] - min_val
        prims_norm[:, channel_idx, ...] = torch.log(F.relu(shifted_data) + EPSILON)
    return prims_norm

def unnormalize_prims(prims_norm, stats, device):
    prims_norm = prims_norm.to(device)
    prims_physical = torch.clone(prims_norm)
    
    mean = stats['mean'].view(-1, 1, 1, 1)
    std = stats['std'].view(-1, 1, 1, 1)
    prims_physical[:, STD_CHANNELS, ...] = (prims_norm[:, STD_CHANNELS, ...] * (std + EPSILON)) + mean
    
    for i, channel_idx in enumerate(LOG_CHANNELS):
        min_val = stats['log_mins'][i]
        norm_data_clamped = torch.clamp(prims_norm[:, channel_idx, ...], max=CLIP_VAL)
        prims_physical[:, channel_idx, ...] = torch.exp(norm_data_clamped) - EPSILON + min_val
    return prims_physical

class SimpleVAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(SimpleVAE, self).__init__()
        
        self.in_channels = input_shape[0]
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(self.in_channels, 32, kernel_size=3, stride=2, padding=1), # D/2
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # D/4
            nn.BatchNorm3d(64), 
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), # D/8
            nn.BatchNorm3d(128), 
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1), # D/16
            nn.BatchNorm3d(256), 
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=3, stride=(3, 2, 2), padding=1),
            nn.BatchNorm3d(512), 
            nn.LeakyReLU(0.2),
        )
        
        self._calculate_conv_output_size(input_shape)
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)
        
        # decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=(3, 2, 2), padding=1, output_padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def _calculate_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.encoder(dummy_input)
            self.unflattened_shape = dummy_output.shape[1:] 
            self.flattened_size = int(np.prod(self.unflattened_shape))
            print(f"VAE dynamically initialized:")
            print(f"  Input shape: {input_shape}")
            print(f"  Encoder output shape: {self.unflattened_shape}")
            print(f"  Flattened features: {self.flattened_size}")

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, max=CLIP_VAL)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        
        # get params for latent space
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # sample
        z = self.reparameterize(mu, logvar)
        
        # decode
        h_decode = self.decoder_input(z)
        h_decode = h_decode.view(-1, *self.unflattened_shape)
        x_recon_norm = self.decoder(h_decode)

        x_recon_norm = F.interpolate(x_recon_norm, size=x.shape[2:])
        
        return x_recon_norm, mu, logvar


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

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cuda'):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous loss: {best_loss:.4e}")
    
    return start_epoch, best_loss


def main():
    torch.autograd.set_detect_anomaly(True) 
    
    print(f"Using device: {DEVICE}")
    
    try:
        prims_files = sorted(glob.glob(os.path.join(DATA_DIR, "primitives", "prims_*.npy")))
        if not prims_files:
            raise FileNotFoundError(f"No 'prims_*.npy' files found")
        
        sample_prims = np.load(prims_files[0]).squeeze(1)
        input_shape = sample_prims.shape
        del sample_prims
        
        # calc norm stats
        stats_dataset = HamrDataset(data_dir=DATA_DIR)
        stats_loader = DataLoader(stats_dataset, 
                                  batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
        norm_stats = get_normalization_stats(stats_loader, device=DEVICE)
        
        train_dataset = HamrDataset(data_dir=DATA_DIR)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                  shuffle=True, num_workers=4, pin_memory=True)
        
    except Exception as e:
        print(f"Failed to load initial data: {e}")
        return
        
    print("Initializing")
    model = SimpleVAE(input_shape=input_shape, 
                      latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 1
    best_loss = float('inf')
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint):
        response = input(f"Found existing checkpoint. Resume training? (y/n): ")
        if response.lower() == 'y':
            start_epoch, best_loss = load_checkpoint(
                latest_checkpoint, model, optimizer, scaler, DEVICE
            )
    
    print(f"starting training")
    print(f"effective batch size: {EFFECTIVE_BATCH_SIZE} ({BATCH_SIZE} * {ACCUMULATION_STEPS} steps)")
    print(f"input shape: {input_shape}")
    print(f"latent dim: {LATENT_DIM}")

    # training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        
        optimizer.zero_grad() 
        
        progress_bar = tqdm(enumerate(train_loader), 
                            desc=f"Epoch {epoch}/{EPOCHS}", 
                            total=len(train_loader))
        
        for batch_idx, prims_data in progress_bar:
            prims_data = prims_data.to(DEVICE)
            
            # forward pass
            with torch.cuda.amp.autocast():
                prims_norm = normalize_prims(prims_data, norm_stats, DEVICE)
                decoded_norm, mu, logvar = model(prims_norm)
                L_recon = F.mse_loss(decoded_norm, prims_norm, reduction='mean')
                logvar_c = torch.clamp(logvar, max=CLIP_VAL)
                L_KL = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
                L_total = (W_RECON * L_recon) + (BETA * L_KL)

                # nan check, model generated, i havent seen this hit tho
            if not torch.isfinite(L_total):
                print(f"\n---!! WARNING: Non-finite TOTAL LOSS detected: {L_total.item():.4e} on batch {batch_idx} !!---")
                print(f"    L_recon: {L_recon.item():.4e}, L_KL: {L_KL.item():.4e}")
                print("    Skipping optimizer step to prevent model corruption.")
                optimizer.zero_grad() # Clear gradients
                continue # Skip this batch
            
            # loss scaling
            L_total_scaled = L_total / ACCUMULATION_STEPS

            # backprop (this is where the model fails as soon as physics loss is added due to nan/inf)
            scaler.scale(L_total_scaled).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # loss logging
            epoch_losses['total'] += L_total.item() 
            epoch_losses['recon'] += L_recon.item()
            epoch_losses['kl'] += L_KL.item()
            
            progress_bar.set_postfix(
                Loss=f"{(epoch_losses['total'] / (batch_idx + 1)):.4e}",
                Recon=f"{(epoch_losses['recon'] / (batch_idx + 1)):.4e}",
                KL=f"{(epoch_losses['kl'] / (batch_idx + 1)):.4e}"
            )

        n_batches = len(train_loader)
        avg_total_loss = epoch_losses['total'] / n_batches
        
        print(f"avg total loss: {avg_total_loss:.4e}")
        print(f"L_Recon:  {epoch_losses['recon'] / n_batches:.4e}")
        print(f"L_KL:  {epoch_losses['kl'] / n_batches:.4e} (Weighted: {BETA * epoch_losses['kl'] / n_batches:.4e})")
        
        is_best = avg_total_loss < best_loss
        if is_best:
            best_loss = avg_total_loss
        
        if epoch % SAVE_EVERY_N_EPOCHS == 0 or is_best or epoch == EPOCHS:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                loss=avg_total_loss,
                checkpoint_dir=CHECKPOINT_DIR,
                is_best=is_best
            )

    print("training ended")
    print(f" best loss achieved: {best_loss:.4e}")

if __name__ == "__main__":
    main()
