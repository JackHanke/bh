import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

# 
class VAE(nn.Module):
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
            logger.info(f"VAE dynamically initialized:")
            logger.info(f"  Input shape: {input_shape}")
            logger.info(f"  Encoder output shape: {self.unflattened_shape}")
            logger.info(f"  Flattened features: {self.flattened_size}")

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
    logger.info(f"checkpoint saved at {checkpoint_path}")
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"best model saved at {best_path}")
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cuda'):
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Resuming from epoch {start_epoch}")
    logger.info(f"Previous loss: {best_loss:.4e}")
    
    return start_epoch, best_loss
