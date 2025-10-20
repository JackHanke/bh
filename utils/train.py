import os
import time
import torch
from torch.utils.data import DataLoader

from dataset import HDF5Dataset

# Create the dataset instance
dataset = HDF5Dataset(DATA_PATH)

# Create the DataLoader
# This is the magic part!
num_workers = os.cpu_count()  # Use all available CPU cores
print(f"\nTraining with {num_workers} workers...")


# num_workers > 0 tells DataLoader to use multiprocessing.
# Each worker process will be an independent HDF5Dataset instance.
# persistent_workers=True keeps the workers alive (and their file handles open)
# between epochs, which is much faster.
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # Automatically handles random sampling
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True  # Speeds up CPU-to-GPU memory copies
)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    start_time = time.time()
    
    for i, (batch_features, batch_labels) in enumerate(train_loader):
        # At this point, the batch is already loaded and pre-processed.
        # Your main loop just does the model forward/backward pass.
        # 'batch_features' and 'batch_labels' are on the CPU.
        # You would now move them to the GPU.
        # e.g., batch_features = batch_features.to('cuda')
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}, Shape: {batch_features.shape}, "
                  f"Time: {time.time() - start_time:.4f}s")
            start_time = time.time()

