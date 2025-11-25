import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HamrDataset(Dataset):
    """
    Loads the prims
    """
    def __init__(self, data_dir):
        self.prims_dir = os.path.join(data_dir, "primitives")
        
        # Find all primitive files
        self.prims_files = sorted(glob.glob(os.path.join(self.prims_dir, "prims_*.npy")))
        if not self.prims_files:
            raise FileNotFoundError(f"prims_*.npy files not found ")
        
        print(f"found {len(self.prims_files)} primitive files.")

    def __len__(self):
        return len(self.prims_files)

    def __getitem__(self, idx):
        prims_path = self.prims_files[idx]
        prims_data = np.load(prims_path).squeeze(1).astype(np.float32)
        prims_data = np.nan_to_num(prims_data, nan=0.0, posinf=0.0, neginf=0.0)
        return prims_data

class HDF5Dataset(Dataset):
    """
    A custom PyTorch Dataset for reading from an HDF5 file.
    
    The key for parallel loading is that each worker process
    will instantiate its own copy of this class.
    """
    def __init__(self, h5_path, percentage: float, dataset_type: str, features_key='data', labels_key='labels'):
        self.h5_path = h5_path
        self.features_key = features_key
        self.labels_key = labels_key
        self.percentage = percentage
        self.dataset_type = dataset_type
        self.size = 3600 # TODO this needs to be fixed
        
        # We will open the file handle *within* __getitem__
        # or, more efficiently, store it here, but it must be
        # initialized *after* the worker forks.
        self.file = None 

    def __len__(self):
        # We can open the file once just to get the length
        with h5py.File(self.h5_path, 'r') as f:
            train_size = int(len(f[self.features_key]) * self.percentage)
            assert self.size == train_size, "I hard coded this, "
            return train_size
            
    def __getitem__(self, index):
        # This check is crucial.
        # Each worker process will have its own 'self.file = None'
        # when it starts, so it will open its *own* file handle.
        # This avoids sharing file handles across processes, which is unsafe.
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        # Get the random slice
        offset = (self.size if self.dataset_type == 'valid' else 0)
        x = self.file[self.features_key][index + offset]
        # y = self.file[self.labels_key][index + offset]
        
        # Apply any transforms here (e.g., convert to tensor)
        x_tensor = torch.from_numpy(x.astype(np.float32))
        # y_tensor = torch.tensor(y, dtype=torch.long)
        
        # return x_tensor, y_tensor
        return x_tensor, x_tensor
