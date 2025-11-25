#
import os
import numpy as np
from tqdm import tqdm
import yaml
#
import logging
import pickle
import yaml
#
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#
from models.vaes import VAE

def main():
    torch.autograd.set_detect_anomaly(True) 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_DIR = "/storage/scratch1/4/skumar680/5000run/data_output/"

    EPOCHS = 25
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
    LATENT_DIM = 256

    CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
    SAVE_EVERY_N_EPOCHS = 3
    SAVE_BEST = True
    
    print(f"Using device: {DEVICE}")
    
    print("Initializing")

    model = VAE(
        input_shape=input_shape, 
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
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
    for epoch in range(EPOCHS):
        model.train()
        
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        
        optimizer.zero_grad() 
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            desc=f"Epoch {epoch}/{EPOCHS}", 
            total=len(train_loader)
        )
        
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
